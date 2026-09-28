# Getting started

## Install

SPARC needs Python 3.9 or later and PyTorch 2.4 or later. Install it from a clone, because the
package runs the model and training code in the repository:

```bash
git clone git@github.com:tommyfuu/SPARC.git
cd SPARC
pip install -e .            # add ".[parquet]" to read Parquet tables
```

This installs the `sparc` command. `sparc --help` lists its subcommands.

## Try it on synthetic data

These commands run on a laptop CPU in a few minutes and write about 1.3 GB, mostly checkpoints. They make a synthetic cohort, pretrain
on it, finetune for neutropenia after CISPLATIN, and evaluate the finetuned model. Run them in
an empty folder outside the repository; they write `data/` and `runs/` there.

```bash
mkdir sparc-demo && cd sparc-demo
sparc synthetic data/DEMO_CAIA              # 200 synthetic patients
sparc validate data/DEMO_CAIA

COMMON="-ds DEMO_CAIA -vp data -m parsimonious -f 1 -e 2 -bs 16 -umt True"
sparc train $COMMON -s pretrain -o pretrain -op runs/pretrain
sparc train $COMMON -s finetune -o neutropenia_risk -me CISPLATIN -an True -lo hybrid \
    -pp runs/pretrain -op runs/neutropenia
sparc eval  $COMMON -o neutropenia_risk -me CISPLATIN -an True -lo hybrid -op runs/neutropenia
```

The last command writes `runs/neutropenia/fold0/test_predictions.csv`, with a risk score for
each test-set infusion whose outcome is known. See [Training and inference](training.md) for
what each flag does.

## Use your own data

1. Export your cohort as the tables in [Input data](data.md) (SPARC_dep package/v1 tables work
   too), into a folder `<data root>/<SITE>_CAIA/`.
2. Check them with `sparc validate <data root>/<SITE>_CAIA`.
3. Copy `examples/pretrain.yaml` and `examples/finetune.yaml`, set the dataset and paths, and
   follow [Training and inference](training.md).
