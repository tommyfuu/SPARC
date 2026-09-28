# Changelog

## 1.0.0 (unreleased)

- `sparc_clinical` package and `sparc` command (`train`, `eval`, `convert`, `validate`,
  `synthetic`), installed with `pip install -e .`. `train` and `eval` run
  `model_train/SPARC_train_inf_risk_fixed.py` and accept YAML configs.
- Input data as JSON, pickle, CSV or Parquet (`data/bundle_io.py`), including SPARC_dep
  package/v1 tables. `load_dataset` reads any of them for `*_CAIA` datasets; the MSK_CAIA files
  are read as before. `sparc validate` also warns about unusual units.
- Training script:
  - New `-bs/--batch_size` (default 512, as before) and `-acc/--accelerator` (default `auto`;
    was always `gpu`). Runs use one device; training on several crashed.
  - Supervised runs write `test_predictions.csv`, scored in the same shuffled batches as the
    test metrics.
  - `-mo test` stops when the fold has no trained checkpoint. Before, with `-pp` set, it tested
    the pretrained model with an untrained head.
- Tests, a documentation site, example configs and SLURM scripts.
