from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from checkpoint_utils import (  # noqa: E402
    MODEL_REVISIONS,
    convert_to_current_format,
    create_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)


def metadata() -> dict:
    model_name = "TencentGameMate/chinese-hubert-base"
    return {
        "checkpoint_kind": "partial_finetune",
        "state_scope": "trainable_overlay",
        "model_name": model_name,
        "model_revision": MODEL_REVISIONS[model_name],
        "pooling": "global",
        "base_vocabulary": ["a", "ai"],
        "architecture": {"dropout": 0.2},
    }


class CheckpointUtilsTest(unittest.TestCase):
    def test_format_one_round_trip(self) -> None:
        state_dict = {"head.weight": torch.arange(6).reshape(2, 3)}
        checkpoint = create_checkpoint(
            state_dict=state_dict,
            metadata=metadata(),
            metrics={"validation": {"base_accuracy": 0.5}},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.pt"
            save_checkpoint(path, checkpoint)
            loaded = load_checkpoint(path)

        self.assertEqual(loaded["format"], 1)
        self.assertEqual(loaded["metadata"], metadata())
        self.assertTrue(
            torch.equal(loaded["state_dict"]["head.weight"], state_dict["head.weight"])
        )

    def test_format_zero_is_converted(self) -> None:
        legacy = {
            "format": 0,
            "trainable_state_dict": {"head.weight": torch.ones(2, 3)},
            "metrics": {
                "model_name": "TencentGameMate/chinese-hubert-base",
                "pooling": "global",
                "base_vocabulary": ["a", "ai"],
                "history": [],
                "validation": {},
            },
        }
        converted = convert_to_current_format(legacy)
        self.assertEqual(converted["format"], 1)
        self.assertEqual(converted["metadata"]["state_scope"], "trainable_overlay")
        self.assertEqual(converted["metrics"], {"history": [], "validation": {}})

    def test_unknown_format_is_rejected(self) -> None:
        checkpoint = {"format": 9, "state_dict": {}, "metrics": {}}
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
            validate_checkpoint(checkpoint)

    def test_format_one_requires_exact_top_level_keys(self) -> None:
        checkpoint = create_checkpoint(state_dict={}, metadata=metadata(), metrics={})
        checkpoint["extra"] = True
        with self.assertRaisesRegex(ValueError, "keys must be exactly"):
            validate_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
