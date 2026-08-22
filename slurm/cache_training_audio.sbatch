#!/bin/bash
#SBATCH --job-name=mandarin-audio-cache
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-audio-cache-%j.out
#SBATCH --error=slurm-audio-cache-%j.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" scripts/cache_training_audio.py --output data/audio_16khz.pt
