from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from train_syllable_classifier import split_training_speaker  # noqa: E402


class TrainingSplitTest(unittest.TestCase):
    def test_stratified_split_retains_each_class_in_training(self) -> None:
        paths = ["a1", "a2", "b1", "b2", "only1", "external"]
        bases = ["a", "a", "b", "b", "only", "a"]
        groups = ["abe", "abe", "abe", "abe", "abe", "yue"]

        training, validation = split_training_speaker(
            paths,
            bases,
            groups,
            train_speaker="abe",
            strategy="stratified-base",
            fraction=0.15,
            seed=7,
        )

        self.assertEqual({bases[index] for index in validation}, {"a", "b"})
        self.assertEqual({bases[index] for index in training}, {"a", "b", "only"})
        self.assertNotIn(5, training)
        self.assertNotIn(5, validation)


if __name__ == "__main__":
    unittest.main()
