#!/usr/bin/env python3
"""Run one classifier checkpoint on an audio recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from checkpoint_utils import load_checkpoint
from extract_speech_features import decode_audio, pool_hidden
from train_syllable_classifier import Classifier
from train_unfrozen_classifier import FineTuneModel, unfreeze_top_layers
from transformers import AutoModel


def apply_trainable_overlay(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Apply a partial state dictionary and reject unintended key differences."""
    expected_overlay = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    provided = set(state_dict)
    missing_overlay = expected_overlay.difference(provided)
    unexpected_overlay = provided.difference(expected_overlay)
    if missing_overlay or unexpected_overlay:
        raise ValueError(
            "Invalid trainable overlay: "
            f"missing={sorted(missing_overlay)}, "
            f"unexpected={sorted(unexpected_overlay)}"
        )

    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected model keys: {result.unexpected_keys}")
    missing_trainable = expected_overlay.intersection(result.missing_keys)
    if missing_trainable:
        raise ValueError(f"Missing trainable model keys: {sorted(missing_trainable)}")


def load_encoder(metadata: dict, model_cache: Path, allow_download: bool):
    """Load the exact pretrained encoder revision named by checkpoint metadata."""
    return AutoModel.from_pretrained(
        metadata["model_name"],
        revision=metadata["model_revision"],
        cache_dir=model_cache,
        local_files_only=not allow_download,
    )


def predict_partial_finetune(
    checkpoint: dict,
    audio: torch.Tensor,
    device: torch.device,
    model_cache: Path,
    allow_download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a partially fine-tuned model and return its logits."""
    metadata = checkpoint["metadata"]
    encoder = load_encoder(metadata, model_cache, allow_download)
    unfreeze_top_layers(encoder, metadata["unfreeze_layers"])
    model = FineTuneModel(
        encoder=encoder,
        pooling=metadata["pooling"],
        bases=len(metadata["base_vocabulary"]),
        dropout=metadata["architecture"]["dropout"],
    )
    apply_trainable_overlay(model, checkpoint["state_dict"])
    model.to(device).eval()

    values = audio.unsqueeze(0).to(device)
    mask = torch.ones_like(values, dtype=torch.long)
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ),
    ):
        return model(values, mask)


def predict_frozen_classifier(
    checkpoint: dict,
    audio: torch.Tensor,
    device: torch.device,
    model_cache: Path,
    allow_download: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a frozen encoder and its trained classifier heads."""
    metadata = checkpoint["metadata"]
    encoder = load_encoder(metadata, model_cache, allow_download).to(device).eval()
    values = audio.unsqueeze(0).to(device)
    mask = torch.ones_like(values, dtype=torch.long)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
    ):
        hidden = encoder(input_values=values, attention_mask=mask).last_hidden_state
    lengths = encoder._get_feat_extract_output_lengths(mask.sum(1)).cpu()
    global_features, temporal_features = pool_hidden(hidden.cpu(), lengths)
    features = global_features if metadata["pooling"] == "global" else temporal_features

    classifier = Classifier(
        shape=tuple(features.shape[1:]),
        pooling=metadata["pooling"],
        bases=len(metadata["base_vocabulary"]),
        dropout=metadata["architecture"]["dropout"],
    )
    classifier.load_state_dict(checkpoint["state_dict"], strict=True)
    classifier.to(device).eval()
    with torch.inference_mode():
        return classifier(features.to(device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    metadata = checkpoint["metadata"]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    audio = decode_audio(args.audio, args.ffmpeg)
    predictors = {
        "partial_finetune": predict_partial_finetune,
        "frozen_encoder_classifier": predict_frozen_classifier,
    }
    predictor = predictors[metadata["checkpoint_kind"]]
    base_logits, tone_logits = predictor(
        checkpoint, audio, device, args.model_cache, args.allow_download
    )

    base_probabilities = base_logits.softmax(-1)[0].float().cpu()
    tone_probabilities = tone_logits.softmax(-1)[0].float().cpu()
    top_count = min(args.top_k, base_probabilities.numel())
    probabilities, indices = base_probabilities.topk(top_count)
    base_predictions = [
        {
            "label": metadata["base_vocabulary"][int(index)],
            "probability": float(probability),
        }
        for probability, index in zip(probabilities, indices)
    ]
    base = base_predictions[0]["label"]
    tone = int(tone_probabilities.argmax()) + 1
    result = {
        "checkpoint": str(args.checkpoint),
        "audio": str(args.audio),
        "model_name": metadata["model_name"],
        "model_revision": metadata["model_revision"],
        "base": base,
        "tone": tone,
        "joint": f"{base}{tone}",
        "base_top_k": base_predictions,
        "tone_probabilities": {
            str(index + 1): float(probability)
            for index, probability in enumerate(tone_probabilities)
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
