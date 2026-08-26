#!/usr/bin/env python3
"""Exercise attention poolers with the cached HuBERT model on one GPU batch."""

from __future__ import annotations

from pathlib import Path

import torch
from checkpoint_utils import MODEL_REVISIONS
from extract_speech_features import MODEL_NAMES
from train_unfrozen_classifier import AudioCollator, FineTuneModel, unfreeze_top_layers
from transformers import AutoModel


def main() -> None:
    device = torch.device("cuda")
    audio = torch.load("data/audio_16khz.pt", map_location="cpu", weights_only=False)
    endpoints = torch.load(
        "data/audio_16khz_endpointed.pt", map_location="cpu", weights_only=False
    )
    items = [
        (audio["waveforms"][index], torch.tensor(0), torch.tensor(0), index)
        for index in range(2)
    ]
    values, mask, _, _, _ = AudioCollator(
        augment=True, endpoint_metadata=endpoints["endpoint_metadata"]
    )(items)
    model_name = MODEL_NAMES["hubert"]
    for pooling in ("attentive_global", "ordered8", "attentive_combined"):
        encoder = AutoModel.from_pretrained(
            model_name,
            revision=MODEL_REVISIONS[model_name],
            cache_dir=Path("models/huggingface"),
            local_files_only=True,
        )
        unfreeze_top_layers(encoder, 4)
        model = FineTuneModel(encoder, pooling, bases=411, dropout=0.2).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base, tone, auxiliary = model(
                values.to(device), mask.to(device), return_auxiliary=True
            )
            loss = base.square().mean() + tone.square().mean()
            if "diversity_loss" in auxiliary:
                loss = loss + auxiliary["diversity_loss"] + auxiliary["ordering_loss"]
        loss.backward()
        print(
            pooling,
            tuple(base.shape),
            tuple(tone.shape),
            float(loss.detach()),
            flush=True,
        )
        del model, encoder
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
