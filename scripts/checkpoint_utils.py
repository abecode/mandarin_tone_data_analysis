"""Versioned checkpoint helpers shared by the training and inference code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT = 0
SUPPORTED_FORMATS = {CHECKPOINT_FORMAT}
STATE_KEYS = {"state_dict", "trainable_state_dict"}


def create_checkpoint(
    *,
    state_key: str,
    state_dict: dict[str, torch.Tensor],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Create a format-0 checkpoint using one supported state-dictionary key."""
    if state_key not in STATE_KEYS:
        choices = ", ".join(sorted(STATE_KEYS))
        raise ValueError(f"state_key must be one of: {choices}")
    return {
        "format": CHECKPOINT_FORMAT,
        state_key: state_dict,
        "metrics": metrics,
    }


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate the common format and top-level checkpoint structure."""
    checkpoint_format = checkpoint.get("format", CHECKPOINT_FORMAT)
    if checkpoint_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format!r}")

    present_state_keys = STATE_KEYS.intersection(checkpoint)
    if len(present_state_keys) != 1:
        raise ValueError(
            "Checkpoint must contain exactly one state dictionary; found "
            f"{sorted(present_state_keys)}"
        )
    if not isinstance(checkpoint.get("metrics"), dict):
        raise TypeError("Checkpoint 'metrics' must be a dictionary")


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted project checkpoint onto CPU and validate its structure."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    validate_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Validate and atomically save a checkpoint.

    The temporary file is written beside the destination so that replacement is
    atomic on the project filesystem. An interrupted write leaves the previous
    checkpoint intact.
    """
    validate_checkpoint(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
