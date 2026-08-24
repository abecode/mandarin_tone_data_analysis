#!/usr/bin/env python3
"""Decode experiment audio once for repeated end-to-end fine-tuning runs."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from extract_speech_features import decode_audio
from speech_endpointing import EndpointConfig, detect_endpoints


def decode_and_endpoint(
    row: dict[str, str], ffmpeg: Path, endpoint: bool
) -> tuple[torch.Tensor, dict[str, float | int | bool | str]]:
    """Decode one recording and optionally retain its detected speech span."""
    waveform = decode_audio(Path(row["path"]), ffmpeg)
    if not endpoint:
        return waveform, {}
    config = EndpointConfig()
    result = detect_endpoints(waveform, config)
    metadata: dict[str, float | int | bool | str] = {
        "original_samples": waveform.numel(),
        "start_sample": result.start_sample,
        "end_sample": result.end_sample,
        "retained_samples": result.end_sample - result.start_sample,
        "noise_floor_db": result.noise_floor_db,
        "threshold_db": result.threshold_db,
        "peak_db": result.peak_db,
        "fallback": result.fallback,
        "fallback_reason": result.fallback_reason,
    }
    return waveform[result.start_sample : result.end_sample].clone(), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/audio_16khz.pt"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    parser.add_argument("--endpoint", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["include_experiment"] == "yes"
        ]
    waveforms = []
    endpoint_metadata = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        decoded = executor.map(
            lambda row: decode_and_endpoint(row, args.ffmpeg, args.endpoint), rows
        )
        for index, (waveform, metadata) in enumerate(decoded, 1):
            waveforms.append(waveform)
            endpoint_metadata.append(metadata)
            if index % 100 == 0 or index == len(rows):
                print(f"{index}/{len(rows)}", flush=True)
    artifact = {
        "format": 1,
        "paths": [row["path"] for row in rows],
        "bases": [row["canonical_base"] for row in rows],
        "tones": [row["canonical_tone"] for row in rows],
        "datasets": [row["dataset"] for row in rows],
        "sample_rate": 16000,
        "waveforms": waveforms,
        "preprocessing": {
            "endpointing": args.endpoint,
            "endpoint_config": EndpointConfig().to_dict() if args.endpoint else None,
        },
        "endpoint_metadata": endpoint_metadata if args.endpoint else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"Wrote {len(rows)} decoded recordings to {args.output}")


if __name__ == "__main__":
    main()
