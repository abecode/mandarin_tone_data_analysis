#!/bin/bash
#SBATCH --job-name=mandarin-frozen-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-frozen-grid-%A_%a.out
#SBATCH --error=slurm-frozen-grid-%A_%a.err

set -euo pipefail

PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
ENCODERS=(hubert xlsr)
ENCODER="${ENCODERS[$SLURM_ARRAY_TASK_ID]}"
FEATURES="$PROJECT_ROOT/data/features/$ENCODER.pt"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/results/frozen_grid}"

cd "$PROJECT_ROOT"
export HF_HOME="$PROJECT_ROOT/models/huggingface"

"$PROJECT_PYTHON" -c \
    'import sys, torch, transformers; print(f"Python: {sys.executable}"); print(f"Torch: {torch.__version__}"); print(f"Transformers: {transformers.__version__}"); print(f"CUDA: {torch.cuda.is_available()}"); assert torch.cuda.is_available()'

if [[ ! -s "$FEATURES" ]]; then
    "$PROJECT_PYTHON" scripts/extract_speech_features.py \
        --encoder "$ENCODER" \
        --output "$FEATURES" \
        --device cuda
fi

for TRAIN_SPEAKER in abe yue; do
    for POOLING in global temporal8; do
        OUTPUT_DIR="$OUTPUT_ROOT/${ENCODER}_${TRAIN_SPEAKER}_${POOLING}"
        "$PROJECT_PYTHON" scripts/train_syllable_classifier.py \
            --config configs/frozen_classifier.json \
            --features "$FEATURES" \
            --train-speaker "$TRAIN_SPEAKER" \
            --pooling "$POOLING" \
            --output-dir "$OUTPUT_DIR" \
            --device cuda
    done
done
