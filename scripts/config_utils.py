"""Small JSON configuration helpers for command-line experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def requested_config_path(default: Path) -> Path:
    """Read only --config from argv before constructing the complete parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=default)
    known, _ = parser.parse_known_args(sys.argv[1:])
    return known.config


def apply_config_defaults(
    parser: argparse.ArgumentParser, path: Path
) -> dict[str, Any]:
    """Load JSON values and use them as argparse defaults."""
    values = json.loads(path.read_text(encoding="utf-8"))
    valid_destinations = {action.dest for action in parser._actions}
    unknown = set(values).difference(valid_destinations)
    if unknown:
        raise ValueError(f"Unknown configuration keys in {path}: {sorted(unknown)}")
    parser.set_defaults(**values)
    return values
