"""Versioned checkpoint helpers shared by training and inference code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CURRENT_CHECKPOINT_FORMAT = 1
SUPPORTED_FORMATS = {0, CURRENT_CHECKPOINT_FORMAT}
LEGACY_STATE_KEYS = {"state_dict", "trainable_state_dict"}
METRIC_KEYS = {"history", "validation", "external", "oli"}
MODEL_REVISIONS = {
    "TencentGameMate/chinese-hubert-base": "fce0375452b1dd6c080ac3248d423d4d037bc831",
    "facebook/wav2vec2-xls-r-300m": "1a640f32ac3e39899438a2931f9924c02f080a54",
}


def split_run_record(
    run_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate reconstruction metadata from measured results."""
    metadata = {
        key: value for key, value in run_record.items() if key not in METRIC_KEYS
    }
    metrics = {key: run_record[key] for key in METRIC_KEYS if key in run_record}
    return metadata, metrics


def create_checkpoint(
    *,
    state_dict: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Create a checkpoint using the current schema."""
    checkpoint = {
        "format": CURRENT_CHECKPOINT_FORMAT,
        "state_dict": state_dict,
        "metadata": metadata,
        "metrics": metrics,
    }
    validate_checkpoint(checkpoint)
    return checkpoint


def validate_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate a checkpoint without changing its format."""
    checkpoint_format = checkpoint.get("format", 0)
    if checkpoint_format not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format!r}")

    if checkpoint_format == 0:
        state_keys = LEGACY_STATE_KEYS.intersection(checkpoint)
        if len(state_keys) != 1:
            raise ValueError(
                "Format-0 checkpoint must contain exactly one state dictionary; "
                f"found {sorted(state_keys)}"
            )
        if not isinstance(checkpoint.get("metrics"), dict):
            raise TypeError("Format-0 checkpoint 'metrics' must be a dictionary")
        return

    expected_keys = {"format", "state_dict", "metadata", "metrics"}
    if set(checkpoint) != expected_keys:
        raise ValueError(
            "Format-1 checkpoint keys must be exactly "
            f"{sorted(expected_keys)}; found {sorted(checkpoint)}"
        )
    if not isinstance(checkpoint["state_dict"], dict):
        raise TypeError("Checkpoint 'state_dict' must be a dictionary")
    if not isinstance(checkpoint["metadata"], dict):
        raise TypeError("Checkpoint 'metadata' must be a dictionary")
    if not isinstance(checkpoint["metrics"], dict):
        raise TypeError("Checkpoint 'metrics' must be a dictionary")

    metadata = checkpoint["metadata"]
    required_metadata = {
        "checkpoint_kind",
        "state_scope",
        "model_name",
        "model_revision",
        "pooling",
        "base_vocabulary",
        "architecture",
    }
    missing = required_metadata.difference(metadata)
    if missing:
        raise ValueError(f"Checkpoint metadata is missing: {sorted(missing)}")
    if metadata["state_scope"] not in {"complete_head", "trainable_overlay"}:
        raise ValueError(f"Unknown state scope: {metadata['state_scope']!r}")


def load_raw_checkpoint(path: Path) -> dict[str, Any]:
    """Load a trusted project checkpoint onto CPU without conversion."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary")
    validate_checkpoint(checkpoint)
    return checkpoint


def convert_to_current_format(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Return a format-1 representation of a supported checkpoint."""
    validate_checkpoint(checkpoint)
    if checkpoint.get("format", 0) == CURRENT_CHECKPOINT_FORMAT:
        return checkpoint

    combined_record = checkpoint["metrics"]
    metadata, metrics = split_run_record(combined_record)
    is_overlay = "trainable_state_dict" in checkpoint
    model_name = metadata["model_name"]
    pooling = metadata["pooling"]
    metadata.update(
        {
            "checkpoint_kind": (
                "partial_finetune" if is_overlay else "frozen_encoder_classifier"
            ),
            "state_scope": "trainable_overlay" if is_overlay else "complete_head",
            "model_revision": MODEL_REVISIONS[model_name],
            "architecture": {
                "dropout": 0.2,
                "projection_size": 256,
                "temporal_bins": 8 if pooling == "temporal8" else None,
                "frame_projection_size": 128 if pooling == "temporal8" else None,
            },
        }
    )
    legacy_state_key = "trainable_state_dict" if is_overlay else "state_dict"
    return create_checkpoint(
        state_dict=checkpoint[legacy_state_key],
        metadata=metadata,
        metrics=metrics,
    )


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load and normalize a trusted project checkpoint to the current format."""
    return convert_to_current_format(load_raw_checkpoint(path))


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Validate and atomically save a checkpoint.

    The temporary file is written beside the destination so replacement is
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
