#!/usr/bin/env python3
"""Build a normalized manifest across the recording filename generations."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


CANONICAL_RE = re.compile(
    r"^(?P<index>\d{4})_(?P<attempt>\d{2})_(?P<label>[a-z]+(?:[1-5])?)_"
    r"(?P<timestamp>\d{8}T\d{6}Z)_(?P<id>[0-9a-f-]{36})$"
)
LEGACY_RE = re.compile(
    r"^(?P<index>\d{4})_(?P<label>[a-z]+(?:[1-5])?)_"
    r"(?:(?:unspecified|row\d+)_)?(?P<base>[a-z]+)_(?P<id>[0-9a-f]{8})$"
)
UUID_RE = re.compile(
    r"^(?P<id>[0-9a-f-]{36})_(?P<label>[a-z]+(?:[1-5])?)$"
)
LABEL_RE = re.compile(r"^(?P<base>[a-z]+?)(?P<tone>[1-5])?$")


FIELDS = [
    "path",
    "dataset",
    "speaker",
    "session",
    "stimulus_index",
    "attempt",
    "label",
    "base_syllable",
    "tone",
    "recorded_at_utc",
    "recording_id",
    "extension",
    "naming_scheme",
    "parse_status",
]


def parse_label(label: str) -> tuple[str, str]:
    match = LABEL_RE.fullmatch(label)
    if not match:
        return "", ""
    return match.group("base"), match.group("tone") or ""


def parse_recording(path: Path, root: Path, dataset: str) -> dict[str, str]:
    relative = path.relative_to(root)
    parts = relative.parts
    # Preserve the uppercase T/Z in canonical UTC timestamps.
    stem = path.stem
    speaker = parts[0] if dataset == "tone_labeled" and len(parts) > 1 else "oli_2"
    session = next((part for part in parts[:-1] if part.startswith("session_")), "")

    values = {field: "" for field in FIELDS}
    values.update(
        path=str(path),
        dataset=dataset,
        speaker=speaker,
        session=session,
        extension=path.suffix.lower(),
    )

    match = CANONICAL_RE.fullmatch(stem)
    if match:
        data = match.groupdict()
        base, tone = parse_label(data["label"])
        values.update(
            stimulus_index=data["index"],
            attempt=data["attempt"],
            label=data["label"],
            base_syllable=base,
            tone=tone,
            recorded_at_utc=data["timestamp"],
            recording_id=data["id"],
            naming_scheme="canonical",
            parse_status="ok",
        )
        return values

    match = LEGACY_RE.fullmatch(stem)
    if match:
        data = match.groupdict()
        base, tone = parse_label(data["label"])
        values.update(
            stimulus_index=data["index"],
            label=data["label"],
            base_syllable=base or data["base"],
            tone=tone,
            recording_id=data["id"],
            naming_scheme="legacy_session",
            parse_status="ok",
        )
        return values

    match = UUID_RE.fullmatch(stem)
    if match:
        data = match.groupdict()
        base, tone = parse_label(data["label"])
        values.update(
            label=data["label"],
            base_syllable=base,
            tone=tone,
            recording_id=data["id"],
            naming_scheme="uuid_label",
            parse_status="ok",
        )
        return values

    values.update(naming_scheme="unknown", parse_status="unparsed")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("raw/mandarin-tone-recordings"))
    parser.add_argument("--output", type=Path, default=Path("data/recordings.csv"))
    args = parser.parse_args()

    sources = [
        (args.raw_root / "audio_fixed_abe", "tone_labeled", {".wav", ".webm"}),
        (args.raw_root / "correct_audio_oli_2", "tone_unspecified", {".wav", ".webm"}),
    ]
    rows: list[dict[str, str]] = []
    for root, dataset, extensions in sources:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions):
            rows.append(parse_recording(path, root, dataset))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    failed = sum(row["parse_status"] != "ok" for row in rows)
    print(f"Wrote {len(rows)} recordings to {args.output} ({failed} unparsed).")


if __name__ == "__main__":
    main()
