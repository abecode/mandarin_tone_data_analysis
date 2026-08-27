#!/usr/bin/env python3
"""Reproduce the Colab frozen-HuBERT, tone-only attention experiment."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from attention_pooling import OrderedAttentionPooling
from checkpoint_utils import MODEL_REVISIONS
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModel

MODEL_ID = "TencentGameMate/chinese-hubert-base"
SPEAKER_1 = "speaker_000000001"
SPEAKER_2 = "speaker_000000002"
SEED = 20260821
SAMPLE_RATE = 16_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("mandarin-isolated-syllables-v0.1/data/metadata.csv"),
    )
    parser.add_argument("--audio-cache", type=Path, default=Path("data/audio_16khz.pt"))
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=Path("data/colab_tone_frame_features.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/colab_tone_only")
    )
    parser.add_argument("--model-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--extraction-batch-size", type=int, default=8)
    parser.add_argument("--training-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["tone"] in {"1", "2", "3", "4"}]


def split_indices(rows: list[dict[str, str]]) -> dict[str, list[int]]:
    speaker_1 = [
        index for index, row in enumerate(rows) if row["speaker_id"] == SPEAKER_1
    ]
    speaker_2 = [
        index for index, row in enumerate(rows) if row["speaker_id"] == SPEAKER_2
    ]
    validation = stratified_limit(speaker_1, rows, round(0.15 * len(speaker_1)), SEED)
    validation_set = set(validation)
    train = [index for index in speaker_1 if index not in validation_set]
    return {"train": train, "validation": validation, "test": speaker_2}


def stratified_limit(
    indices: list[int], rows: list[dict[str, str]], maximum: int, seed: int
) -> list[int]:
    if len(indices) <= maximum:
        return list(indices)
    rng = np.random.default_rng(seed)
    by_tone: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        by_tone[rows[index]["tone"]].append(index)
    exact_counts = {
        tone: maximum * len(values) / len(indices) for tone, values in by_tone.items()
    }
    counts = {tone: int(value) for tone, value in exact_counts.items()}
    remaining = maximum - sum(counts.values())
    priority = sorted(
        by_tone,
        key=lambda tone: (exact_counts[tone] - counts[tone], tone),
        reverse=True,
    )
    for tone in priority[:remaining]:
        counts[tone] += 1
    selected = []
    for tone in sorted(by_tone):
        shuffled = rng.permutation(by_tone[tone]).tolist()
        selected.extend(shuffled[: counts[tone]])
    rng.shuffle(selected)
    return selected


def classroom_indices(
    full: dict[str, list[int]], rows: list[dict[str, str]]
) -> dict[str, list[int]]:
    return {
        "train": stratified_limit(full["train"], rows, 1600, SEED),
        "validation": stratified_limit(full["validation"], rows, 400, SEED + 1),
        "test": stratified_limit(full["test"], rows, 800, SEED + 2),
    }


def add_boundary_silence(
    waveform: torch.Tensor, rng: np.random.Generator
) -> torch.Tensor:
    maximum = SAMPLE_RATE // 2
    before = int(rng.integers(0, maximum + 1))
    after = int(rng.integers(0, maximum + 1))
    rms = float(waveform.float().square().mean().sqrt())

    def boundary(length: int) -> torch.Tensor:
        if rng.random() < 0.5:
            return waveform.new_zeros(length)
        noise = rng.normal(0, max(rms * 0.01, 1e-5), length).astype(np.float32)
        return torch.from_numpy(noise)

    return torch.cat((boundary(before), waveform, boundary(after)))


def extract_batch(
    encoder: nn.Module,
    feature_extractor,
    waveforms: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    inputs = feature_extractor(
        [waveform.numpy() for waveform in waveforms],
        sampling_rate=SAMPLE_RATE,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    values = inputs.input_values.to(device)
    mask = inputs.attention_mask.to(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=device.type == "cuda"),
    ):
        hidden = encoder(input_values=values, attention_mask=mask).last_hidden_state
    lengths = encoder._get_feat_extract_output_lengths(mask.sum(1))
    return [
        sequence[: int(length)].detach().cpu().to(torch.float16)
        for sequence, length in zip(hidden, lengths)
    ]


def build_feature_cache(
    args: argparse.Namespace,
    rows: list[dict[str, str]],
    indices: dict[str, list[int]],
) -> dict[str, object]:
    audio_artifact = torch.load(
        args.audio_cache, map_location="cpu", weights_only=False
    )
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        all_metadata = list(csv.DictReader(handle))
    if len(all_metadata) != len(audio_artifact["waveforms"]):
        raise ValueError("Export metadata and waveform cache lengths differ")
    recording_to_waveform = {
        row["recording_id"]: waveform
        for row, waveform in zip(all_metadata, audio_artifact["waveforms"])
    }
    waveforms = [recording_to_waveform[row["recording_id"]] for row in rows]
    device = torch.device(args.device)
    revision = MODEL_REVISIONS[MODEL_ID]
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID, revision=revision, cache_dir=args.model_cache
    )
    encoder = (
        AutoModel.from_pretrained(
            MODEL_ID, revision=revision, cache_dir=args.model_cache
        )
        .to(device)
        .eval()
    )
    encoder.requires_grad_(False)

    def extract(selected: list[int], augmented: bool = False) -> list[torch.Tensor]:
        output = []
        rng = np.random.default_rng(SEED + 10_000)
        for start in range(0, len(selected), args.extraction_batch_size):
            batch_indices = selected[start : start + args.extraction_batch_size]
            batch = [waveforms[index] for index in batch_indices]
            if augmented:
                batch = [add_boundary_silence(waveform, rng) for waveform in batch]
            output.extend(extract_batch(encoder, feature_extractor, batch, device))
            if len(output) % 200 < args.extraction_batch_size:
                print(
                    f"Extracted {'augmented ' if augmented else ''}"
                    f"{len(output)}/{len(selected)}",
                    flush=True,
                )
        return output

    features = {
        "format": 0,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "row_recording_ids": [row["recording_id"] for row in rows],
        "indices": indices,
        "original": {split: extract(selected) for split, selected in indices.items()},
        "augmented_train": extract(indices["train"], augmented=True),
    }
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, args.feature_cache)
    return features


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        indices: list[int],
        rows: list[dict[str, str]],
        original: list[torch.Tensor],
        augmented: list[torch.Tensor] | None = None,
    ) -> None:
        self.labels = torch.tensor(
            [int(rows[index]["tone"]) - 1 for index in indices], dtype=torch.long
        )
        self.original = original
        self.augmented = augmented

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.augmented[index] if self.augmented else self.original[index]
        return features, self.labels[index]


def collate_features(
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features, labels = zip(*items)
    lengths = torch.tensor([len(feature) for feature in features])
    padded = nn.utils.rnn.pad_sequence(
        [feature.float() for feature in features], batch_first=True
    )
    return padded, lengths, torch.stack(labels)


class ToneModel(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.pooling = OrderedAttentionPooling(hidden_size, dropout=dropout)
        self.project = nn.Sequential(
            nn.LayerNorm(8 * 128),
            nn.Linear(8 * 128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tone_head = nn.Linear(256, 4)

    def forward(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        summary, auxiliary = self.pooling(hidden, lengths)
        return self.tone_head(self.project(summary)), auxiliary


def make_loader(
    indices: list[int],
    rows: list[dict[str, str]],
    original: list[torch.Tensor],
    batch_size: int,
    augmented: list[torch.Tensor] | None = None,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        CachedFeatureDataset(indices, rows, original, augmented),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_features,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, object]:
    model.eval()
    predictions = []
    targets = []
    for hidden, lengths, labels in loader:
        logits, _ = model(hidden.to(device), lengths.to(device))
        predictions.extend(logits.argmax(1).cpu().tolist())
        targets.extend(labels.tolist())
    matrix = np.zeros((4, 4), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    return {
        "accuracy": sum(
            target == prediction for target, prediction in zip(targets, predictions)
        )
        / len(targets),
        "count": len(targets),
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalized.tolist(),
    }


def train_condition(
    args: argparse.Namespace,
    scope_name: str,
    condition: str,
    rows: list[dict[str, str]],
    indices: dict[str, list[int]],
    features: dict[str, object],
) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    random.seed(SEED)
    model = ToneModel(768).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    full_positions = {
        split: {
            index: position for position, index in enumerate(features["indices"][split])
        }
        for split in ("train", "validation", "test")
    }

    def select_features(split: str) -> list[torch.Tensor]:
        return [
            features["original"][split][full_positions[split][index]]
            for index in indices[split]
        ]

    original = {split: select_features(split) for split in indices}
    augmented = None
    if condition == "augmented":
        augmented = [
            features["augmented_train"][full_positions["train"][index]]
            for index in indices["train"]
        ]
    loaders = {
        "train": make_loader(
            indices["train"],
            rows,
            original["train"],
            args.training_batch_size,
            augmented=augmented,
            shuffle=True,
        ),
        "validation": make_loader(
            indices["validation"],
            rows,
            original["validation"],
            args.training_batch_size,
        ),
        "test": make_loader(
            indices["test"],
            rows,
            original["test"],
            args.training_batch_size,
        ),
    }
    best_accuracy = -1.0
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = defaultdict(list)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for hidden, lengths, labels in loaders["train"]:
            hidden = hidden.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(hidden, lengths)
            loss = (
                F.cross_entropy(logits, labels)
                + 0.01 * auxiliary["diversity_loss"]
                + 0.01 * auxiliary["ordering_loss"]
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        validation = evaluate(model, loaders["validation"], device)
        history["loss"].append(float(np.mean(losses)))
        history["validation_accuracy"].append(validation["accuracy"])
        print(
            f"{scope_name}/{condition}: epoch={epoch:02d} "
            f"loss={np.mean(losses):.4f} validation={validation['accuracy']:.4f}",
            flush=True,
        )
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    model.load_state_dict(best_state)
    output = {
        "scope": scope_name,
        "condition": condition,
        "best_epoch": best_epoch,
        "history": dict(history),
        "validation": evaluate(model, loaders["validation"], device),
        "test": evaluate(model, loaders["test"], device),
        "split_counts": {split: len(values) for split, values in indices.items()},
        "tone_counts": {
            split: dict(Counter(rows[index]["tone"] for index in split_indices))
            for split, split_indices in indices.items()
        },
    }
    run_dir = args.output_dir / f"{scope_name}_{condition}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": 0,
            "model_state_dict": best_state,
            "metadata": output,
        },
        run_dir / "classifier.pt",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    args = parse_args()
    rows = load_metadata(args.metadata)
    full = split_indices(rows)
    classroom = classroom_indices(full, rows)
    if args.feature_cache.exists() and not args.rebuild_feature_cache:
        features = torch.load(
            args.feature_cache, map_location="cpu", weights_only=False
        )
        if features["row_recording_ids"] != [row["recording_id"] for row in rows]:
            raise ValueError("Feature cache recording order does not match metadata")
    else:
        features = build_feature_cache(args, rows, full)
    results = []
    for scope_name, indices in (("classroom", classroom), ("full", full)):
        for condition in ("original", "augmented"):
            results.append(
                train_condition(args, scope_name, condition, rows, indices, features)
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
