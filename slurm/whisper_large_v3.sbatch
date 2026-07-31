#!/bin/bash
#SBATCH --job-name=whisper-large-v3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-whisper-large-v3-%j.out
#SBATCH --error=slurm-whisper-large-v3-%j.err

set -euo pipefail

# Activate the project environment before submitting, or customize this line.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    source "$VIRTUAL_ENV/bin/activate"
fi

python scripts/transcribe_whisper.py \
    --manifest data/recordings.csv \
    --output results/whisper_large_v3.jsonl \
    --model large-v3 \
    --device cuda
