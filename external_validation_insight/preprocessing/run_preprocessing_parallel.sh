#!/bin/bash
# Same steps as run_preprocessing.sh, but the six pipelines run as separate processes at the same time.
# Each writes only its own pickles, so they do not collide. Wall time ~ the slowest pipeline (labs) instead of the sum.
#   nohup bash run_preprocessing_parallel.sh > preprocessing.log 2>&1 &
# Per-pipeline logs: $INSIGHT_OUTPUT_DIR/logs/<pipeline>.log
set -euo pipefail
export PYTHONUNBUFFERED=1

# defaults: sparc_paths.INSIGHT_RAW_DIR / INSIGHT_DATA_DIR (see config.py); override with these variables
CODE=$(cd "$(dirname "$0")" && pwd)
export INSIGHT_DATA_DIR=${INSIGHT_DATA_DIR:-$(cd "$CODE" && python -c "import config; print(config.DATA_DIR)")}
export INSIGHT_OUTPUT_DIR=${INSIGHT_OUTPUT_DIR:-$(cd "$CODE" && python -c "import config; print(config.OUTPUT_DIR)")}

LOGS=$INSIGHT_OUTPUT_DIR/logs
mkdir -p "$LOGS"
cd "$CODE"
echo "start: $(date)"

python preflight_check.py --data-dir "$INSIGHT_DATA_DIR"

# one process per pipeline group (oral and non-oral meds come from the same pass)
declare -A PIDS
for group in labs vitals "oral_meds nonoral_meds" admissions stage demographics; do
    name=${group%% *}; [ "$name" = oral_meds ] && name=meds
    python run_preprocessing_split.py --run-preprocessing --pipelines $group \
        --data-dir "$INSIGHT_DATA_DIR" --output-dir "$INSIGHT_OUTPUT_DIR" > "$LOGS/$name.log" 2>&1 &
    PIDS[$name]=$!
    echo "started $name (pid ${PIDS[$name]}) -> $LOGS/$name.log"
done

fail=0
for name in "${!PIDS[@]}"; do
    if wait "${PIDS[$name]}"; then echo "finished $name  $(date)"; else echo "FAILED $name (see $LOGS/$name.log)"; fail=1; fi
done
[ $fail -eq 0 ] || { echo "stopping: a pipeline failed"; exit 1; }

python make_views.py --output-dir "$INSIGHT_OUTPUT_DIR"
python create_splits.py
echo "done: $(date)"
