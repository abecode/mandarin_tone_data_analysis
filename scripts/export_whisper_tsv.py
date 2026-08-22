#!/usr/bin/env python3
"""Export Whisper JSONL results as a compact, segment-level TSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import TextIO

FIELDS = [
    "fname",
    "label",
    "asr_text",
    "start",
    "end",
    "avg_logprob",
    "no_speech_prob",
]


def output_handle(path: Path | None) -> tuple[TextIO, bool]:
    if path is None:
        return sys.stdout, False
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", newline="", encoding="utf-8"), True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert transcribe_whisper.py JSONL output to segment-level TSV."
    )
    parser.add_argument("input", type=Path, help="Whisper JSONL result file")
    parser.add_argument("-o", "--output", type=Path, help="Output TSV; default: stdout")
    parser.add_argument(
        "--full-path",
        action="store_true",
        help="Write the full recording path instead of only its filename",
    )
    args = parser.parse_args()

    handle, should_close = output_handle(args.output)
    try:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()

        with args.input.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON on {args.input}:{line_number}: {exc}"
                    ) from exc

                recording_path = result.get("path", "")
                fname = recording_path if args.full_path else Path(recording_path).name
                shared = {
                    "fname": fname,
                    "label": result.get("label", ""),
                    "asr_text": result.get("asr_text", ""),
                }
                segments = result.get("segments") or []
                if not segments:
                    writer.writerow(shared)
                    continue

                for segment in segments:
                    writer.writerow(
                        {
                            **shared,
                            "start": segment.get("start", ""),
                            "end": segment.get("end", ""),
                            "avg_logprob": segment.get("avg_logprob", ""),
                            "no_speech_prob": segment.get("no_speech_prob", ""),
                        }
                    )
    finally:
        if should_close:
            handle.close()


if __name__ == "__main__":
    main()
