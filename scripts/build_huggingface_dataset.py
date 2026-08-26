#!/usr/bin/env python3
"""Build the private Mandarin isolated-syllable Hugging Face dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

SPEAKER_BY_DATASET = {
    "tone_labeled": "speaker_000000001",
    "abe_new": "speaker_000000001",
    "tone_labeled_yue": "speaker_000000002",
    "tone_unspecified": "speaker_000000003",
}
SOURCE_COLLECTION = {
    "tone_labeled": "source_collection_000000001",
    "abe_new": "source_collection_000000002",
    "tone_labeled_yue": "source_collection_000000003",
    "tone_unspecified": "source_collection_000000004",
}
SPEAKER_BACKGROUND = {
    "speaker_000000001": "learner",
    "speaker_000000002": "native",
    "speaker_000000003": "learner",
}
MANDARIN_LEVEL = {
    "speaker_000000001": "intermediate",
    "speaker_000000002": "",
    "speaker_000000003": "beginner",
}
SPEAKER_LANGUAGES = [
    {
        "speaker_id": "speaker_000000001",
        "language_tag": "en",
        "relationship": "native",
        "proficiency": "native_like",
        "other_language_name": "",
        "sort_order": 1,
    },
    {
        "speaker_id": "speaker_000000001",
        "language_tag": "cmn-Hans-CN",
        "relationship": "learned",
        "proficiency": "intermediate",
        "other_language_name": "",
        "sort_order": 2,
    },
    {
        "speaker_id": "speaker_000000002",
        "language_tag": "cmn-Hans-CN",
        "relationship": "native",
        "proficiency": "native_like",
        "other_language_name": "",
        "sort_order": 1,
    },
    {
        "speaker_id": "speaker_000000003",
        "language_tag": "en",
        "relationship": "native",
        "proficiency": "native_like",
        "other_language_name": "",
        "sort_order": 1,
    },
    {
        "speaker_id": "speaker_000000003",
        "language_tag": "cmn-Hans-CN",
        "relationship": "learned",
        "proficiency": "beginner",
        "other_language_name": "",
        "sort_order": 2,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--audio-cache", type=Path, default=Path("data/audio_16khz.pt"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    parser.add_argument(
        "--output", type=Path, default=Path("mandarin-isolated-syllables-v0.1")
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def session_key(row: dict[str, str]) -> str:
    """Construct an internal grouping key that will not be published."""
    if row["session"]:
        source_session = row["session"]
    else:
        source_session = str(Path(row["path"]).parent)
    return "|".join((row["dataset"], row["speaker"], source_session))


def encode_flac(
    waveform: torch.Tensor, destination: Path, ffmpeg: Path
) -> tuple[int, str]:
    """Encode a cached 16-kHz mono waveform as lossless FLAC."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".flac.tmp")
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        "flac",
        "-compression_level",
        "8",
        "-f",
        "flac",
        "-y",
        str(temporary),
    ]
    try:
        subprocess.run(
            command,
            input=waveform.detach().float().numpy().tobytes(),
            check=True,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return destination.stat().st_size, digest


def write_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = [
            row for row in csv.DictReader(handle) if row["include_experiment"] == "yes"
        ]
    artifact = torch.load(args.audio_cache, map_location="cpu", weights_only=False)
    manifest_paths = [row["path"] for row in manifest]
    if manifest_paths != artifact["paths"]:
        raise ValueError(
            "Manifest and decoded-audio cache contain different path orders"
        )
    unknown = set(row["dataset"] for row in manifest).difference(SPEAKER_BY_DATASET)
    if unknown:
        raise ValueError(f"Unknown source datasets: {sorted(unknown)}")

    session_ids = {}
    for speaker_id in sorted(set(SPEAKER_BY_DATASET.values())):
        keys = sorted(
            {
                session_key(row)
                for row in manifest
                if SPEAKER_BY_DATASET[row["dataset"]] == speaker_id
            }
        )
        session_ids.update(
            {
                key: f"{speaker_id}_session_{index:06d}"
                for index, key in enumerate(keys, 1)
            }
        )

    export_rows = []
    encoding_jobs = []
    for index, (row, waveform) in enumerate(zip(manifest, artifact["waveforms"]), 1):
        recording_id = f"recording_{index:09d}"
        speaker_id = SPEAKER_BY_DATASET[row["dataset"]]
        relative_audio = Path("audio") / speaker_id / f"{recording_id}.flac"
        destination = args.output / "data" / relative_audio
        encoding_jobs.append((waveform, destination, args.ffmpeg))
        tone = row["canonical_tone"]
        export_rows.append(
            {
                "file_name": relative_audio.as_posix(),
                "recording_id": recording_id,
                "speaker_id": speaker_id,
                "session_id": session_ids[session_key(row)],
                "speaker_background": SPEAKER_BACKGROUND[speaker_id],
                "source_collection": SOURCE_COLLECTION[row["dataset"]],
                "prompt_label": row["canonical_label"],
                "base_syllable": row["canonical_base"],
                "tone": tone,
                "tone_specified": str(tone in {"1", "2", "3", "4", "5"}).lower(),
                "stimulus_index": row["stimulus_index"],
                "attempt": row["attempt"],
                "duration_seconds": f"{waveform.numel() / 16000:.6f}",
                "sample_rate": 16000,
                "original_format": row["extension"].lstrip("."),
                "naming_scheme": row["naming_scheme"],
            }
        )

    encoded = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(
            executor.map(lambda values: encode_flac(*values), encoding_jobs), 1
        ):
            encoded.append(result)
            if index % 100 == 0 or index == len(encoding_jobs):
                print(f"Encoded {index}/{len(encoding_jobs)}", flush=True)

    write_table(args.output / "data" / "metadata.csv", export_rows)
    speaker_rows = [
        {
            "speaker_id": speaker_id,
            "knows_mandarin": "true",
            "speaker_background": SPEAKER_BACKGROUND[speaker_id],
            "mandarin_level": MANDARIN_LEVEL[speaker_id],
            "recordings": sum(row["speaker_id"] == speaker_id for row in export_rows),
        }
        for speaker_id in sorted(SPEAKER_BACKGROUND)
    ]
    write_table(args.output / "speakers.csv", speaker_rows)
    write_table(args.output / "speaker_languages.csv", SPEAKER_LANGUAGES)
    total_bytes = sum(size for size, _ in encoded)
    release = {
        "dataset_name": "mandarin-isolated-syllables",
        "version": "0.1.0",
        "recordings": len(export_rows),
        "audio_bytes": total_bytes,
        "sample_rate": 16000,
        "audio_format": "FLAC",
        "speaker_counts": dict(Counter(row["speaker_id"] for row in export_rows)),
        "tone_counts": dict(
            Counter(row["tone"] or "unspecified" for row in export_rows)
        ),
    }
    (args.output / "release.json").write_text(
        json.dumps(release, indent=2) + "\n", encoding="utf-8"
    )

    checksum_paths = [
        args.output / "README.md",
        args.output / "DATA_USE_TERMS.md",
        args.output / "speakers.csv",
        args.output / "speaker_languages.csv",
        args.output / "release.json",
        args.output / "data" / "metadata.csv",
    ]
    checksum_paths.extend(destination for _, destination, _ in encoding_jobs)
    checksum_file = args.output / "SHA256SUMS"
    with checksum_file.open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            relative = path.relative_to(args.output)
            handle.write(f"{file_digest(path)}  {relative.as_posix()}\n")
    print(json.dumps(release, indent=2))
    print(f"Wrote dataset to {args.output}")


if __name__ == "__main__":
    main()
