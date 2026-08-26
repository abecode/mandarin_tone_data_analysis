"""Boundary-silence augmentation for untrimmed isolated speech."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SilenceAugmentationConfig:
    """Parameters for randomized leading and trailing nonspeech."""

    sample_rate: int = 16_000
    maximum_ms: int = 500
    zero_probability: float = 0.25
    noise_probability: float = 0.25


def _repeat_to_length(source: torch.Tensor, length: int) -> torch.Tensor:
    if length == 0:
        return source.new_empty(0)
    if source.numel() == 0:
        return source.new_zeros(length)
    repeats = (length + source.numel() - 1) // source.numel()
    return source.repeat(repeats)[:length]


def _boundary_audio(
    waveform: torch.Tensor,
    metadata: dict,
    length: int,
    leading: bool,
    config: SilenceAugmentationConfig,
) -> torch.Tensor:
    draw = random.random()
    if draw < config.zero_probability:
        return waveform.new_zeros(length)
    if leading:
        source = waveform[: int(metadata["start_sample"])]
    else:
        source = waveform[int(metadata["end_sample"]) :]
    if draw < config.zero_probability + config.noise_probability or not source.numel():
        boundary = waveform[: min(waveform.numel(), config.sample_rate // 10)]
        rms = boundary.square().mean().sqrt().clamp_min(1e-5)
        return torch.randn(length, dtype=waveform.dtype) * rms
    if source.numel() > length:
        start = random.randint(0, source.numel() - length)
        return source[start : start + length]
    return _repeat_to_length(source, length)


def add_boundary_silence(
    waveform: torch.Tensor,
    metadata: dict,
    config: SilenceAugmentationConfig = SilenceAugmentationConfig(),
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Add independently sampled valid nonspeech to both waveform boundaries."""
    maximum = round(config.sample_rate * config.maximum_ms / 1_000)
    leading_length = random.randint(0, maximum)
    trailing_length = random.randint(0, maximum)
    leading = _boundary_audio(waveform, metadata, leading_length, True, config)
    trailing = _boundary_audio(waveform, metadata, trailing_length, False, config)
    return torch.cat((leading, waveform, trailing)), (leading_length, trailing_length)
