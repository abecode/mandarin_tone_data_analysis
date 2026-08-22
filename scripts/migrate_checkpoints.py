#!/usr/bin/env python3
"""Migrate classifier checkpoints to the current schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_utils import (
    CURRENT_CHECKPOINT_FORMAT,
    convert_to_current_format,
    load_raw_checkpoint,
    save_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "roots",
        type=Path,
        nargs="*",
        default=[Path("results")],
        help="Files or directories to search (default: results)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite checkpoints atomically; otherwise perform a dry run",
    )
    args = parser.parse_args()

    paths: set[Path] = set()
    for root in args.roots:
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(root.rglob("classifier.pt"))

    changed = 0
    for path in sorted(paths):
        checkpoint = load_raw_checkpoint(path)
        old_format = checkpoint.get("format", 0)
        if old_format == CURRENT_CHECKPOINT_FORMAT:
            print(f"unchanged format={old_format}: {path}")
            continue

        action = "migrate" if args.apply else "would migrate"
        print(f"{action} format {old_format} -> {CURRENT_CHECKPOINT_FORMAT}: {path}")
        if args.apply:
            migrated = convert_to_current_format(checkpoint)
            save_checkpoint(path, migrated)
            reloaded = load_raw_checkpoint(path)
            if reloaded["format"] != CURRENT_CHECKPOINT_FORMAT:
                raise RuntimeError(f"Migration verification failed: {path}")
        changed += 1

    action = "Migrated" if args.apply else "Would migrate"
    print(f"{action} {changed} of {len(paths)} checkpoints.")


if __name__ == "__main__":
    main()
