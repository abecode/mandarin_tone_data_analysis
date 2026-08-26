"""Tests for learned utterance pooling and boundary-silence augmentation."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from attention_pooling import (  # noqa: E402
    AttentiveStatisticsPooling,
    OrderedAttentionPooling,
)
from silence_augmentation import (  # noqa: E402
    SilenceAugmentationConfig,
    add_boundary_silence,
)


class AttentionPoolingTests(unittest.TestCase):
    def test_attentive_statistics_ignores_padding(self) -> None:
        pooling = AttentiveStatisticsPooling(4)
        hidden = torch.randn(2, 10, 4)
        lengths = torch.tensor([6, 10])

        summary, attention = pooling(hidden, lengths)

        self.assertEqual(summary.shape, (2, 8))
        self.assertTrue(torch.allclose(attention.sum(1), torch.ones(2)))
        self.assertEqual(attention[0, 6:].count_nonzero(), 0)

    def test_ordered_attention_produces_ordered_initial_centers(self) -> None:
        pooling = OrderedAttentionPooling(16, dropout=0.0)
        hidden = torch.randn(3, 30, 16)

        summary, auxiliary = pooling(hidden, torch.tensor([30, 24, 18]))

        self.assertEqual(summary.shape, (3, 1024))
        centers = auxiliary["centers"]
        self.assertTrue(torch.all(centers[:, 1:] > centers[:, :-1]))
        self.assertTrue(torch.isfinite(auxiliary["diversity_loss"]))
        self.assertTrue(torch.isfinite(auxiliary["ordering_loss"]))

    def test_silence_augmentation_preserves_original_waveform(self) -> None:
        random.seed(7)
        waveform = torch.arange(100, dtype=torch.float32)
        config = SilenceAugmentationConfig(sample_rate=1_000, maximum_ms=20)

        augmented, (leading, trailing) = add_boundary_silence(
            waveform,
            {"start_sample": 10, "end_sample": 90},
            config,
        )

        self.assertTrue(torch.equal(augmented[leading : leading + 100], waveform))
        self.assertEqual(augmented.numel(), leading + 100 + trailing)


if __name__ == "__main__":
    unittest.main()
