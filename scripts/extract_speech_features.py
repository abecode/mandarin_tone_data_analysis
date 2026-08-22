#!/usr/bin/env python3
"""Extract frozen speech-encoder features for all experiment recordings."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel

MODEL_NAMES = {
    "hubert": "TencentGameMate/chinese-hubert-base",
    "xlsr": "facebook/wav2vec2-xls-r-300m",
}


def decode_audio(path: Path, ffmpeg: Path) -> torch.Tensor:
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-i",
        str(path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if audio.size == 0:
        raise ValueError(f"Decoded empty audio: {path}")
    return torch.from_numpy(audio)


def collate_audio(
    items: list[tuple[dict[str, str], torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(audio.numel() for _, audio in items)
    values = torch.zeros(len(items), length, dtype=torch.float32)
    mask = torch.zeros(len(items), length, dtype=torch.long)
    for index, (_, audio) in enumerate(items):
        values[index, : audio.numel()] = audio
        mask[index, : audio.numel()] = 1
    return values, mask


def pool_hidden(
    hidden: torch.Tensor, output_lengths: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    global_features = []
    temporal_features = []
    for sequence, length in zip(hidden, output_lengths.tolist()):
        sequence = sequence[: max(1, int(length))].float()
        mean = sequence.mean(dim=0)
        std = sequence.std(dim=0, unbiased=False)
        global_features.append(torch.cat((mean, std)))
        # Adaptive pooling partitions relative syllable time into eight ordered regions.
        temporal = F.adaptive_avg_pool1d(sequence.T.unsqueeze(0), 8).squeeze(0).T
        temporal_features.append(temporal)
    return torch.stack(global_features), torch.stack(temporal_features)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output = args.output or Path(f"data/features/{args.encoder}.pt")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["include_experiment"] == "yes"
        ]
    if args.limit is not None:
        rows = rows[: args.limit]

    device = torch.device(args.device)
    model_name = MODEL_NAMES[args.encoder]
    model = (
        AutoModel.from_pretrained(model_name, cache_dir=args.model_cache)
        .to(device)
        .eval()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    paths: list[str] = []
    bases: list[str] = []
    tones: list[str] = []
    datasets: list[str] = []
    global_batches: list[torch.Tensor] = []
    temporal_batches: list[torch.Tensor] = []

    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        decoded = [
            (row, decode_audio(Path(row["path"]), args.ffmpeg)) for row in batch_rows
        ]
        values, input_mask = collate_audio(decoded)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
        ):
            hidden = model(
                input_values=values.to(device), attention_mask=input_mask.to(device)
            ).last_hidden_state
        if hasattr(model, "_get_feat_extract_output_lengths"):
            lengths = model._get_feat_extract_output_lengths(
                input_mask.sum(dim=1)
            ).cpu()
        else:
            lengths = torch.full((len(batch_rows),), hidden.shape[1], dtype=torch.long)
        global_features, temporal_features = pool_hidden(hidden.cpu(), lengths)
        global_batches.append(global_features.half())
        temporal_batches.append(temporal_features.half())
        paths.extend(row["path"] for row in batch_rows)
        bases.extend(row["canonical_base"] for row in batch_rows)
        tones.extend(row["canonical_tone"] for row in batch_rows)
        datasets.extend(row["dataset"] for row in batch_rows)
        print(f"{min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    artifact = {
        "encoder": args.encoder,
        "model_name": model_name,
        "paths": paths,
        "bases": bases,
        "tones": tones,
        "datasets": datasets,
        "global": torch.cat(global_batches),
        "temporal8": torch.cat(temporal_batches),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    print(f"Wrote {len(paths)} recordings to {output}")


if __name__ == "__main__":
    main()
