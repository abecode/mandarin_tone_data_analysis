#!/bin/bash
#SBATCH --job-name=mandarin-colab-tone
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-colab-tone-%j.out
#SBATCH --error=slurm-colab-tone-%j.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
cd "$PROJECT_ROOT"
"$PROJECT_ROOT/.venv/bin/python" scripts/run_colab_tone_experiment.py \
    --device cuda
