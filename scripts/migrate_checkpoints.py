#!/usr/bin/env python3
"""Add the format-0 marker to legacy classifier checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from checkpoint_utils import CHECKPOINT_FORMAT, load_checkpoint, save_checkpoint


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
        checkpoint = load_checkpoint(path)
        if "format" in checkpoint:
            print(f"unchanged format={checkpoint['format']}: {path}")
            continue
        print(f"{'migrate' if args.apply else 'would migrate'} to format 0: {path}")
        if args.apply:
            checkpoint["format"] = CHECKPOINT_FORMAT
            save_checkpoint(path, checkpoint)
            reloaded = load_checkpoint(path)
            if reloaded["format"] != CHECKPOINT_FORMAT:
                raise RuntimeError(f"Migration verification failed: {path}")
        changed += 1

    action = "Migrated" if args.apply else "Would migrate"
    print(f"{action} {changed} of {len(paths)} checkpoints.")


if __name__ == "__main__":
    main()
