#!/usr/bin/env python3
"""Audit WebM decoding against container durations and report decoder warnings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def stable_sample(paths: list[Path], size: int, seed: int) -> list[Path]:
    return sorted(
        paths, key=lambda path: hashlib.sha256(f"{seed}:{path}".encode()).digest()
    )[:size]


def probe_duration(path: Path, ffprobe: Path) -> float:
    result = subprocess.run(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(json.loads(result.stdout)["format"]["duration"])


def decode(path: Path, ffmpeg: Path) -> tuple[float, str]:
    result = subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "warning",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    samples = len(result.stdout) // 2
    return samples / 16000.0, result.stderr.decode("utf-8", errors="replace").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    parser.add_argument("--ffprobe", type=Path, default=Path("models/linux/ffprobe"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output", type=Path, default=Path("results/webm_decode_audit.tsv")
    )
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["include_experiment"] == "yes"
        ]
    webm = [Path(row["path"]) for row in rows if row["extension"] == ".webm"]
    selected = stable_sample(webm, min(args.sample_size, len(webm)), args.seed)
    audit = []
    for path in selected:
        container = probe_duration(path, args.ffprobe)
        decoded, warning = decode(path, args.ffmpeg)
        audit.append(
            {
                "path": str(path),
                "container_seconds": f"{container:.6f}",
                "decoded_seconds": f"{decoded:.6f}",
                "delta_seconds": f"{decoded - container:.6f}",
                "warning": warning.replace("\t", " ").replace("\n", " | "),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=audit[0], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(audit)
    absolute_deltas = [abs(float(row["delta_seconds"])) for row in audit]
    print(
        json.dumps(
            {
                "sample_size": len(audit),
                "warnings": sum(bool(row["warning"]) for row in audit),
                "max_absolute_duration_delta_seconds": max(absolute_deltas),
                "mean_absolute_duration_delta_seconds": sum(absolute_deltas)
                / len(absolute_deltas),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
