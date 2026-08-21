#!/bin/bash
#SBATCH --job-name=whisper-large-v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-whisper-large-v3-%j.out
#SBATCH --error=slurm-whisper-large-v3-%j.err

set -euo pipefail

PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
cd "$PROJECT_ROOT"

"$PROJECT_PYTHON" -c \
    'import sys, torch; print(f"Python: {sys.executable}"); print(f"Torch: {torch.__version__}"); print(f"CUDA available: {torch.cuda.is_available()}"); assert torch.cuda.is_available(), "Slurm did not expose an allocated CUDA GPU"'

"$PROJECT_PYTHON" scripts/transcribe_whisper.py \
    --manifest data/recordings.csv \
    --output results/whisper_large_v3.jsonl \
    --model large-v3 \
    --device cuda
