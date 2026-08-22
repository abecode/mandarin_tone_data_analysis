#!/usr/bin/env python3
"""Decode experiment audio once for repeated end-to-end fine-tuning runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from extract_speech_features import decode_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/audio_16khz.pt"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["include_experiment"] == "yes"
        ]
    waveforms = []
    for index, row in enumerate(rows, 1):
        waveforms.append(decode_audio(Path(row["path"]), args.ffmpeg))
        if index % 100 == 0 or index == len(rows):
            print(f"{index}/{len(rows)}", flush=True)
    artifact = {
        "paths": [row["path"] for row in rows],
        "bases": [row["canonical_base"] for row in rows],
        "tones": [row["canonical_tone"] for row in rows],
        "datasets": [row["dataset"] for row in rows],
        "sample_rate": 16000,
        "waveforms": waveforms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    print(f"Wrote {len(rows)} decoded recordings to {args.output}")


if __name__ == "__main__":
    main()
