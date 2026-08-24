#!/bin/bash
#SBATCH --job-name=mandarin-endpoint-cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-endpoint-cache-%j.out
#SBATCH --error=slurm-endpoint-cache-%j.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" scripts/cache_training_audio.py \
    --endpoint \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output data/audio_16khz_endpointed.pt
