#!/bin/bash
#SBATCH --job-name=mandarin-endpoint-hubert
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-endpoint-hubert-%A_%a.out
#SBATCH --error=slurm-endpoint-hubert-%A_%a.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
TRAIN_SPEAKERS=(abe yue)
TRAIN_SPEAKER="${TRAIN_SPEAKERS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT_ROOT"

"$PROJECT_PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
for POOLING in global temporal8 temporal16; do
    OUTPUT_DIR="$PROJECT_ROOT/results/unfrozen_grid_endpointed/hubert_${TRAIN_SPEAKER}_${POOLING}"
    "$PROJECT_PYTHON" scripts/train_unfrozen_classifier.py \
        --config configs/unfrozen_extended.json \
        --audio-cache data/audio_16khz_endpointed.pt \
        --encoder hubert \
        --train-speaker "$TRAIN_SPEAKER" \
        --pooling "$POOLING" \
        --output-dir "$OUTPUT_DIR" \
        --device cuda
done
