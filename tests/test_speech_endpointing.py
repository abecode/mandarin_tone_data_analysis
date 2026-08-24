"""Tests for conservative isolated-syllable endpoint detection."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from speech_endpointing import EndpointConfig, detect_endpoints  # noqa: E402


class SpeechEndpointingTests(unittest.TestCase):
    def test_detects_tone_with_silence_and_margin(self) -> None:
        sample_rate = 16_000
        silence = torch.zeros(sample_rate // 2)
        time = torch.arange(sample_rate // 2) / sample_rate
        speech = 0.2 * torch.sin(2 * torch.pi * 200 * time)
        waveform = torch.cat((silence, speech, silence))

        result = detect_endpoints(waveform)

        self.assertFalse(result.fallback)
        self.assertAlmostEqual(result.start_sample / sample_rate, 0.42, delta=0.03)
        self.assertAlmostEqual(result.end_sample / sample_rate, 1.08, delta=0.03)

    def test_falls_back_for_silence(self) -> None:
        waveform = torch.zeros(16_000)

        result = detect_endpoints(waveform, EndpointConfig())

        self.assertTrue(result.fallback)
        self.assertEqual(result.start_sample, 0)
        self.assertEqual(result.end_sample, waveform.numel())


if __name__ == "__main__":
    unittest.main()
