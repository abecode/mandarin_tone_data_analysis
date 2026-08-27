#!/usr/bin/env python3
"""Build the single-download archive for the Hugging Face dataset."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

DEFAULT_ROOT = Path("mandarin-isolated-syllables-v0.1")
DEFAULT_ARCHIVE_NAME = "mandarin_isolated_syllables_v0.1.1.zip"
TOP_LEVEL_FILES = (
    "DATA_USE_TERMS.md",
    "release.json",
    "speaker_languages.csv",
    "speakers.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--archive-name", default=DEFAULT_ARCHIVE_NAME)
    return parser.parse_args()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_inputs(root: Path) -> list[Path]:
    top_level = [root / name for name in TOP_LEVEL_FILES]
    missing = [path for path in top_level if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing release files: {missing}")
    files = list(top_level)
    files.extend(sorted((root / "data").rglob("*")))
    files = [path for path in files if path.is_file()]
    return files


def update_checksums(root: Path, archive: Path, digest: str) -> None:
    checksum_path = root / "SHA256SUMS"
    archive_entry = f"{digest}  {archive.name}"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if not line.endswith(f"  {archive.name}")]
    lines.append(archive_entry)
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    archive = root / args.archive_name
    inputs = archive_inputs(root)
    temporary = archive.with_suffix(".zip.tmp")
    try:
        with ZipFile(
            temporary, mode="w", compression=ZIP_STORED, allowZip64=True
        ) as output:
            for index, path in enumerate(inputs, 1):
                output.write(path, path.relative_to(root).as_posix())
                if index % 500 == 0 or index == len(inputs):
                    print(f"Archived {index}/{len(inputs)}", flush=True)
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    digest = file_digest(archive)
    update_checksums(root, archive, digest)
    print(f"Wrote {archive} ({archive.stat().st_size} bytes)")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
