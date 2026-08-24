#!/usr/bin/env python3
"""Run endpoint detection on a small balanced sample and create an SVG audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
from collections import defaultdict
from pathlib import Path

import torch
from extract_speech_features import decode_audio
from speech_endpointing import EndpointConfig, EndpointResult, detect_endpoints


def group_name(row: dict[str, str]) -> str:
    """Map manifest datasets to the three experimental speakers."""
    if row["dataset"] in {"tone_labeled", "abe_new"}:
        return "abe"
    if row["dataset"] == "tone_labeled_yue":
        return "yue"
    return "oli"


def stable_key(row: dict[str, str], seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{row['path']}".encode()).digest()


def balanced_sample(
    rows: list[dict[str, str]], per_group: int, seed: int
) -> list[dict[str, str]]:
    """Sample speakers and, where available, tones evenly."""
    selected = []
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[group_name(row)].append(row)
    for name in ("abe", "yue", "oli"):
        candidates = groups[name]
        candidates.sort(key=lambda row: stable_key(row, seed))
        if name == "oli":
            selected.extend(candidates[:per_group])
            continue
        by_tone: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in candidates:
            if row["canonical_tone"] in {"1", "2", "3", "4"}:
                by_tone[row["canonical_tone"]].append(row)
        quota = max(1, per_group // 4)
        initial = [row for tone in "1234" for row in by_tone[tone][:quota]]
        chosen = {row["path"] for row in initial}
        remainder = [row for row in candidates if row["path"] not in chosen]
        selected.extend((initial + remainder)[:per_group])
    return selected


def waveform_envelope(waveform: torch.Tensor, points: int = 500) -> list[float]:
    """Return a compact absolute-amplitude envelope."""
    audio = waveform.abs()
    bins = min(points, audio.numel())
    return (
        torch.nn.functional.adaptive_max_pool1d(audio.reshape(1, 1, -1), bins)
        .flatten()
        .tolist()
    )


def write_svg(
    output: Path,
    waveform: torch.Tensor,
    result: EndpointResult,
    title: str,
    sample_rate: int,
) -> None:
    """Write a dependency-free waveform and frame-energy visualization."""
    width, height = 900, 280
    left, right = 45, 15
    plot_width = width - left - right
    envelope = waveform_envelope(waveform)
    maximum = max(max(envelope), 1e-6)
    waveform_points = " ".join(
        f"{left + i * plot_width / max(1, len(envelope) - 1):.1f},"
        f"{105 - 75 * value / maximum:.1f}"
        for i, value in enumerate(envelope)
    )
    energies = result.frame_energy_db.tolist()
    low = min(energies + [result.threshold_db]) if energies else -80
    high = max(energies + [result.threshold_db]) if energies else 0

    def energy_y(value: float) -> float:
        return 245 - 95 * (value - low) / max(high - low, 1e-6)

    energy_points = " ".join(
        f"{left + i * plot_width / max(1, len(energies) - 1):.1f},{energy_y(value):.1f}"
        for i, value in enumerate(energies)
    )
    start_x = left + plot_width * result.start_sample / waveform.numel()
    end_x = left + plot_width * result.end_sample / waveform.numel()
    duration = waveform.numel() / sample_rate
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="18" font-family="sans-serif" font-size="13">{html.escape(title)}</text>
<rect x="{start_x:.1f}" y="25" width="{end_x - start_x:.1f}" height="225" fill="#dbeafe"/>
<polyline points="{waveform_points}" fill="none" stroke="#1f2937" stroke-width="1"/>
<line x1="{left}" y1="130" x2="{width - right}" y2="130" stroke="#9ca3af"/>
<polyline points="{energy_points}" fill="none" stroke="#b45309" stroke-width="1.5"/>
<line x1="{left}" y1="{energy_y(result.threshold_db):.1f}" x2="{width - right}" y2="{energy_y(result.threshold_db):.1f}" stroke="#dc2626" stroke-dasharray="5 4"/>
<line x1="{start_x:.1f}" y1="25" x2="{start_x:.1f}" y2="250" stroke="#2563eb"/>
<line x1="{end_x:.1f}" y1="25" x2="{end_x:.1f}" y2="250" stroke="#2563eb"/>
<text x="{left}" y="272" font-family="sans-serif" font-size="11">0 s</text>
<text x="{width - 70}" y="272" font-family="sans-serif" font-size="11">{duration:.2f} s</text>
</svg>
"""
    output.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/recordings.csv"))
    parser.add_argument("--ffmpeg", type=Path, default=Path("models/linux/ffmpeg"))
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/endpointing_audit")
    )
    args = parser.parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle) if row["include_experiment"] == "yes"
        ]
    selected = balanced_sample(rows, args.per_group, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = EndpointConfig()
    audit = []
    for index, row in enumerate(selected, 1):
        waveform = decode_audio(Path(row["path"]), args.ffmpeg)
        result = detect_endpoints(waveform, config)
        original = waveform.numel() / config.sample_rate
        start = result.start_sample / config.sample_rate
        end = result.end_sample / config.sample_rate
        plot_name = f"{index:02d}_{group_name(row)}_{row['canonical_label'] or row['canonical_base']}.svg"
        write_svg(
            args.output_dir / plot_name,
            waveform,
            result,
            f"{group_name(row)} {row['canonical_label']} — {row['path']}",
            config.sample_rate,
        )
        audit.append(
            {
                "group": group_name(row),
                "label": row["canonical_label"],
                "path": row["path"],
                "original_seconds": f"{original:.4f}",
                "start_seconds": f"{start:.4f}",
                "end_seconds": f"{end:.4f}",
                "retained_seconds": f"{end - start:.4f}",
                "trimmed_fraction": f"{1 - (end - start) / original:.4f}",
                "noise_floor_db": f"{result.noise_floor_db:.2f}",
                "threshold_db": f"{result.threshold_db:.2f}",
                "peak_db": f"{result.peak_db:.2f}",
                "fallback": str(result.fallback).lower(),
                "fallback_reason": result.fallback_reason,
                "plot": plot_name,
            }
        )
    output = args.output_dir / "audit.tsv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=audit[0], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(audit)
    print(f"Wrote {len(audit)} audit rows and SVG plots to {args.output_dir}")


if __name__ == "__main__":
    main()
