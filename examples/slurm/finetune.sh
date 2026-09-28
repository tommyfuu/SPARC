#!/bin/bash
# Finetune one fold per array task (folds 1-10). Submit from the repository root:
#     mkdir -p logs && sbatch examples/slurm/finetune.sh
# Edit the partition, environment and CONFIG below for your cluster and data.
#SBATCH --job-name=sparc_finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/sparc_finetune_%A_%a.out

set -euo pipefail
CONFIG=${CONFIG:-examples/finetune.yaml}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SPARC_env            # an environment with `pip install -e .` of this repository

sparc train --config "$CONFIG" -f "$SLURM_ARRAY_TASK_ID" -acc gpu
