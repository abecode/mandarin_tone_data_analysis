#!/usr/bin/env python3
"""Resumably transcribe a recording manifest with OpenAI Whisper."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import static_ffmpeg
import torch
import whisper


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("status") == "ok":
                    done.add(row["path"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/whisper_large_v3.jsonl")
    )
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--model-dir", type=Path, default=Path("models/whisper"))
    parser.add_argument(
        "--dataset",
        action="append",
        help="Manifest dataset to process; repeat for multiple cohorts (default: tone_labeled)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--limit", type=int, help="Transcribe only this many new files (smoke tests)."
    )
    args = parser.parse_args()

    # Whisper invokes `ffmpeg` as a subprocess. Supply a project-independent
    # binary on compute nodes where FFmpeg is not installed system-wide.
    local_ffmpeg = Path("models/linux/ffmpeg")
    if local_ffmpeg.exists():
        os.environ["PATH"] = os.pathsep.join(
            [str(local_ffmpeg.parent.resolve()), os.environ.get("PATH", "")]
        )
    else:
        static_ffmpeg.add_paths()

    selected_datasets = set(args.dataset or ["tone_labeled"])
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["dataset"] in selected_datasets
        ]

    done = completed_ids(args.output)
    pending = [row for row in rows if row["path"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(
        args.model, device=args.device, download_root=str(args.model_dir)
    )
    print(f"Loaded {args.model} on {args.device}; {len(pending)} recordings pending.")

    with args.output.open("a", encoding="utf-8") as handle:
        for number, row in enumerate(pending, start=1):
            try:
                result = model.transcribe(
                    row["path"],
                    language="zh",
                    task="transcribe",
                    temperature=0,
                    condition_on_previous_text=False,
                    fp16=args.device.startswith("cuda"),
                    verbose=False,
                )
                segments = result.get("segments", [])
                output = {
                    **row,
                    "asr_model": args.model,
                    "asr_device": args.device,
                    "asr_language": "zh",
                    "asr_task": "transcribe",
                    "asr_temperature": 0,
                    "asr_text": result.get("text", ""),
                    "detected_language": result.get("language", ""),
                    "segments": segments,
                    "mean_no_speech_prob": (
                        sum(s.get("no_speech_prob", 0.0) for s in segments)
                        / len(segments)
                        if segments
                        else None
                    ),
                    "mean_avg_logprob": (
                        sum(s.get("avg_logprob", 0.0) for s in segments) / len(segments)
                        if segments
                        else None
                    ),
                    "status": "ok",
                    "error": "",
                }
            except (
                Exception
            ) as exc:  # Preserve failures so batch runs remain resumable.
                output = {
                    **row,
                    "asr_model": args.model,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
            handle.flush()
            if number % 25 == 0 or number == len(pending):
                print(f"Completed {number}/{len(pending)} new recordings.")


if __name__ == "__main__":
    main()
