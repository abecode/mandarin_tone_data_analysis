#!/bin/bash
#SBATCH --job-name=mandarin-hf-dataset
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-hf-dataset-%j.out
#SBATCH --error=slurm-hf-dataset-%j.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" scripts/build_huggingface_dataset.py \
    --workers "$SLURM_CPUS_PER_TASK"
