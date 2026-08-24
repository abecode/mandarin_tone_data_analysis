#!/usr/bin/env python3
"""Conservative energy-based endpoint detection for isolated syllables."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class EndpointConfig:
    """Parameters controlling frame-energy endpoint detection."""

    sample_rate: int = 16_000
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    noise_percentile: float = 0.2
    noise_margin_db: float = 8.0
    peak_floor_db: float = -35.0
    minimum_active_ms: float = 40.0
    maximum_gap_ms: float = 100.0
    margin_ms: float = 80.0
    minimum_retained_ms: float = 250.0
    maximum_trim_fraction: float = 0.8

    def to_dict(self) -> dict[str, float | int]:
        """Return a serialization-friendly representation."""
        return asdict(self)


@dataclass(frozen=True)
class EndpointResult:
    """Detected speech span and diagnostic frame energies."""

    start_sample: int
    end_sample: int
    threshold_db: float
    noise_floor_db: float
    peak_db: float
    fallback: bool
    fallback_reason: str
    frame_energy_db: torch.Tensor
    active_frames: torch.Tensor


def _runs(values: torch.Tensor) -> list[tuple[int, int, bool]]:
    """Return half-open runs from a one-dimensional Boolean tensor."""
    if values.numel() == 0:
        return []
    result = []
    start = 0
    current = bool(values[0])
    for index in range(1, values.numel()):
        value = bool(values[index])
        if value != current:
            result.append((start, index, current))
            start = index
            current = value
    result.append((start, values.numel(), current))
    return result


def _clean_activity(
    active: torch.Tensor, minimum_frames: int, maximum_gap_frames: int
) -> torch.Tensor:
    """Remove brief active runs and bridge brief internal gaps."""
    cleaned = active.clone()
    for start, end, value in _runs(cleaned):
        if value and end - start < minimum_frames:
            cleaned[start:end] = False
    runs = _runs(cleaned)
    for run_index, (start, end, value) in enumerate(runs):
        internal = 0 < run_index < len(runs) - 1
        if not value and internal and end - start <= maximum_gap_frames:
            cleaned[start:end] = True
    return cleaned


def detect_endpoints(
    waveform: torch.Tensor, config: EndpointConfig = EndpointConfig()
) -> EndpointResult:
    """Find a conservative speech span using recording-adaptive RMS energy."""
    audio = waveform.detach().float().flatten()
    frame_length = round(config.sample_rate * config.frame_ms / 1_000)
    hop_length = round(config.sample_rate * config.hop_ms / 1_000)
    if audio.numel() < frame_length:
        return _fallback(audio, "shorter_than_one_frame")

    frames = audio.unfold(0, frame_length, hop_length)
    rms = frames.square().mean(dim=1).sqrt().clamp_min(1e-8)
    energy_db = 20 * torch.log10(rms)
    sorted_energy = energy_db.sort().values
    noise_index = min(
        sorted_energy.numel() - 1,
        max(0, round(config.noise_percentile * (sorted_energy.numel() - 1))),
    )
    noise_floor_db = float(sorted_energy[noise_index])
    peak_db = float(energy_db.max())
    threshold_db = max(
        noise_floor_db + config.noise_margin_db,
        peak_db + config.peak_floor_db,
    )
    active = energy_db >= threshold_db
    minimum_frames = max(1, round(config.minimum_active_ms / config.hop_ms))
    maximum_gap_frames = max(0, round(config.maximum_gap_ms / config.hop_ms))
    active = _clean_activity(active, minimum_frames, maximum_gap_frames)
    indices = active.nonzero().flatten()
    if indices.numel() == 0:
        return _fallback(
            audio,
            "no_active_frames",
            energy_db,
            active,
            threshold_db,
            noise_floor_db,
            peak_db,
        )

    margin = round(config.sample_rate * config.margin_ms / 1_000)
    start = max(0, int(indices[0]) * hop_length - margin)
    end = min(audio.numel(), int(indices[-1]) * hop_length + frame_length + margin)
    minimum_samples = round(config.sample_rate * config.minimum_retained_ms / 1_000)
    trim_fraction = 1 - (end - start) / audio.numel()
    if end - start < minimum_samples:
        reason = "retained_span_too_short"
    elif trim_fraction > config.maximum_trim_fraction:
        reason = "trim_fraction_too_large"
    else:
        reason = ""
    if reason:
        return _fallback(
            audio,
            reason,
            energy_db,
            active,
            threshold_db,
            noise_floor_db,
            peak_db,
        )
    return EndpointResult(
        start,
        end,
        threshold_db,
        noise_floor_db,
        peak_db,
        False,
        "",
        energy_db,
        active,
    )


def _fallback(
    audio: torch.Tensor,
    reason: str,
    energy_db: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    threshold_db: float = float("nan"),
    noise_floor_db: float = float("nan"),
    peak_db: float = float("nan"),
) -> EndpointResult:
    """Construct a result retaining the original waveform."""
    if energy_db is None:
        energy_db = torch.empty(0)
    if active is None:
        active = torch.empty(0, dtype=torch.bool)
    return EndpointResult(
        0,
        audio.numel(),
        threshold_db,
        noise_floor_db,
        peak_db,
        True,
        reason,
        energy_db,
        active,
    )
