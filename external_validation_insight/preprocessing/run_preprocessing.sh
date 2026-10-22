#!/bin/bash
# INSIGHT raw PCORnet CSVs -> SPARC per-site pickles, overall/, full-site view and 1:1:1 splits, for one INSIGHT data pull.
# Needs ~120-200 GB RAM and several hours (the lab file is ~50 GB). Run inside tmux/screen or nohup:
#   nohup bash run_preprocessing.sh > preprocessing.log 2>&1 &
set -euo pipefail
export PYTHONUNBUFFERED=1   # progress shows up in the log as it happens

# defaults: sparc_paths.INSIGHT_RAW_DIR / INSIGHT_DATA_DIR (see config.py); override with these variables
CODE=$(cd "$(dirname "$0")" && pwd)
export INSIGHT_DATA_DIR=${INSIGHT_DATA_DIR:-$(cd "$CODE" && python -c "import config; print(config.DATA_DIR)")}
export INSIGHT_OUTPUT_DIR=${INSIGHT_OUTPUT_DIR:-$(cd "$CODE" && python -c "import config; print(config.OUTPUT_DIR)")}


echo "raw:    $INSIGHT_DATA_DIR"
echo "output: $INSIGHT_OUTPUT_DIR"
echo "start:  $(date)"
mkdir -p "$INSIGHT_OUTPUT_DIR"
cd "$CODE"

# 0. inputs present with the expected columns
python preflight_check.py --data-dir "$INSIGHT_DATA_DIR"

# 1. all six pipelines, written per site (A-F)
python run_preprocessing_split.py --run-preprocessing \
    --data-dir "$INSIGHT_DATA_DIR" --output-dir "$INSIGHT_OUTPUT_DIR"

# 2. overall/ and the full-site (zero-shot) view
python make_views.py --output-dir "$INSIGHT_OUTPUT_DIR"

# 3. stratified 1:1:1 train/val/test splits -> insight_stratified/ and insight_pooled_stratified/
python create_splits.py

echo "done: $(date)"
