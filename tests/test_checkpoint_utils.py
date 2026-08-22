from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from checkpoint_utils import (  # noqa: E402
    create_checkpoint,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)


class CheckpointUtilsTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        state_dict = {"head.weight": torch.arange(6).reshape(2, 3)}
        checkpoint = create_checkpoint(
            state_key="trainable_state_dict",
            state_dict=state_dict,
            metrics={"name": "test"},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier.pt"
            save_checkpoint(path, checkpoint)
            loaded = load_checkpoint(path)

        self.assertEqual(loaded["format"], 0)
        self.assertEqual(loaded["metrics"], {"name": "test"})
        self.assertTrue(
            torch.equal(
                loaded["trainable_state_dict"]["head.weight"],
                state_dict["head.weight"],
            )
        )

    def test_legacy_format_zero_is_accepted(self) -> None:
        checkpoint = {"state_dict": {}, "metrics": {}}
        validate_checkpoint(checkpoint)

    def test_unknown_format_is_rejected(self) -> None:
        checkpoint = {"format": 1, "state_dict": {}, "metrics": {}}
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint format"):
            validate_checkpoint(checkpoint)

    def test_exactly_one_state_dictionary_is_required(self) -> None:
        checkpoint = {
            "format": 0,
            "state_dict": {},
            "trainable_state_dict": {},
            "metrics": {},
        }
        with self.assertRaisesRegex(ValueError, "exactly one state dictionary"):
            validate_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
