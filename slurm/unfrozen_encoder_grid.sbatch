#!/bin/bash
#SBATCH --job-name=mandarin-unfrozen-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-unfrozen-grid-%A_%a.out
#SBATCH --error=slurm-unfrozen-grid-%A_%a.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
ENCODERS=(hubert xlsr)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT_ROOT"

"$PROJECT_PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
for TRAIN_SPEAKER in abe yue; do
    for POOLING in global temporal8; do
        OUTPUT_DIR="$PROJECT_ROOT/results/unfrozen_grid/${ENCODER}_${TRAIN_SPEAKER}_${POOLING}"
        "$PROJECT_PYTHON" scripts/train_unfrozen_classifier.py \
            --config configs/unfrozen_classifier.json \
            --audio-cache data/audio_16khz.pt \
            --encoder "$ENCODER" \
            --train-speaker "$TRAIN_SPEAKER" \
            --pooling "$POOLING" \
            --output-dir "$OUTPUT_DIR" \
            --device cuda
    done
done
