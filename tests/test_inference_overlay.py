from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from predict_classifier import apply_trainable_overlay  # noqa: E402
from train_unfrozen_classifier import FineTuneModel  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(2, 2)
        self.head = nn.Linear(2, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=32)


class InferenceOverlayTest(unittest.TestCase):
    def test_temporal_pooling_has_matched_flattened_size(self) -> None:
        temporal8 = FineTuneModel(TinyEncoder(), "temporal8", bases=10, dropout=0.2)
        temporal16 = FineTuneModel(TinyEncoder(), "temporal16", bases=10, dropout=0.2)

        self.assertEqual(temporal8.project[0].normalized_shape, (1024,))
        self.assertEqual(temporal16.project[0].normalized_shape, (1024,))
        self.assertEqual(temporal8.frame_project[1].out_features, 128)
        self.assertEqual(temporal16.frame_project[1].out_features, 64)

    def test_overlay_changes_only_trainable_parameters(self) -> None:
        model = TinyModel()
        frozen_before = model.frozen.weight.detach().clone()
        state = {
            "head.weight": torch.full_like(model.head.weight, 3.0),
            "head.bias": torch.full_like(model.head.bias, 4.0),
        }

        apply_trainable_overlay(model, state)

        self.assertTrue(torch.equal(model.frozen.weight, frozen_before))
        self.assertTrue(torch.equal(model.head.weight, state["head.weight"]))
        self.assertTrue(torch.equal(model.head.bias, state["head.bias"]))

    def test_overlay_rejects_a_missing_trainable_key(self) -> None:
        model = TinyModel()
        with self.assertRaisesRegex(ValueError, "missing=.*head.bias"):
            apply_trainable_overlay(model, {"head.weight": model.head.weight})


if __name__ == "__main__":
    unittest.main()
