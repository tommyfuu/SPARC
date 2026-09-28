#!/bin/bash
# Evaluate one fold per array task (folds 1-10). Submit from the repository root:
#     mkdir -p logs && sbatch examples/slurm/inference.sh
# Edit the partition, environment and CONFIG below for your cluster and data.
#SBATCH --job-name=sparc_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/sparc_inference_%A_%a.out

set -euo pipefail
CONFIG=${CONFIG:-examples/finetune.yaml}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate SPARC_env            # an environment with `pip install -e .` of this repository

sparc eval --config "$CONFIG" -f "$SLURM_ARRAY_TASK_ID" -acc gpu
