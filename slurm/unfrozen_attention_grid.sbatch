#!/bin/bash
#SBATCH --job-name=mandarin-attention
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --array=0-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-attention-%A_%a.out
#SBATCH --error=slurm-attention-%A_%a.err

set -euo pipefail
PROJECT_ROOT=/beegfs/home/kaze7539/mandarin_tones_data_analysis
PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
TRAIN_SPEAKERS=(abe yue)
POOLINGS=(attentive_global ordered8 attentive_combined)
TRAIN_SPEAKER="${TRAIN_SPEAKERS[$SLURM_ARRAY_TASK_ID]}"
cd "$PROJECT_ROOT"

"$PROJECT_PYTHON" -c 'import torch; print(torch.__version__, torch.cuda.is_available()); assert torch.cuda.is_available()'
for AUGMENTATION in none silence; do
    AUGMENTATION_ARGS=()
    if [[ "$AUGMENTATION" == silence ]]; then
        AUGMENTATION_ARGS+=(--silence-augmentation)
    fi
    for POOLING in "${POOLINGS[@]}"; do
        RUN_NAME="hubert_${TRAIN_SPEAKER}_${POOLING}_${AUGMENTATION}"
        OUTPUT_DIR="$PROJECT_ROOT/results/unfrozen_attention_grid/$RUN_NAME"
        "$PROJECT_PYTHON" scripts/train_unfrozen_classifier.py \
            --config configs/unfrozen_attention.json \
            --audio-cache data/audio_16khz.pt \
            --endpoint-cache data/audio_16khz_endpointed.pt \
            --encoder hubert \
            --train-speaker "$TRAIN_SPEAKER" \
            --pooling "$POOLING" \
            --output-dir "$OUTPUT_DIR" \
            --device cuda \
            "${AUGMENTATION_ARGS[@]}"
    done
done
