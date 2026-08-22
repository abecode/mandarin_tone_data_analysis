#!/usr/bin/env python3
"""Train base-syllable and tone heads on cached frozen-encoder features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from checkpoint_utils import (
    MODEL_REVISIONS,
    create_checkpoint,
    save_checkpoint,
    split_run_record,
)
from config_utils import apply_config_defaults, requested_config_path
from torch import nn
from torch.utils.data import DataLoader, Dataset

ABE_DATASETS = {"tone_labeled", "abe_new"}
YUE_DATASETS = {"tone_labeled_yue"}
OLI_DATASETS = {"tone_unspecified"}


def speaker_group(dataset: str) -> str:
    if dataset in ABE_DATASETS:
        return "abe"
    if dataset in YUE_DATASETS:
        return "yue"
    if dataset in OLI_DATASETS:
        return "oli"
    raise ValueError(f"Unknown dataset: {dataset}")


def validation_member(path: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha256(f"{seed}:{path}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return value < fraction


def stable_order(path: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{path}".encode()).digest()


def split_training_speaker(
    paths: list[str],
    bases: list[str],
    groups: list[str],
    train_speaker: str,
    strategy: str,
    fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    candidates = [index for index, group in enumerate(groups) if group == train_speaker]
    if strategy == "hash-fraction":
        validation = [
            index
            for index in candidates
            if validation_member(paths[index], fraction, seed)
        ]
        validation_set = set(validation)
        return [
            index for index in candidates if index not in validation_set
        ], validation

    by_base: dict[str, list[int]] = {}
    for index in candidates:
        by_base.setdefault(bases[index], []).append(index)
    validation = []
    for indices in by_base.values():
        # Never remove the only training example of a class.
        if len(indices) >= 2:
            validation.append(
                min(indices, key=lambda index: stable_order(paths[index], seed))
            )
    validation_set = set(validation)
    return [index for index in candidates if index not in validation_set], sorted(
        validation
    )


class FeatureDataset(Dataset):
    def __init__(
        self,
        features: torch.Tensor,
        base: torch.Tensor,
        tone: torch.Tensor,
        indices: list[int],
    ):
        self.features = features
        self.base = base
        self.tone = tone
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = self.indices[index]
        return self.features[row].float(), self.base[row], self.tone[row], row


class Classifier(nn.Module):
    def __init__(
        self, shape: tuple[int, ...], pooling: str, bases: int, dropout: float
    ):
        super().__init__()
        self.pooling = pooling
        if pooling == "global":
            width = shape[0]
            self.project = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 256),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            width = shape[1]
            self.frame_project = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, 128),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.project = nn.Sequential(
                nn.LayerNorm(8 * 128),
                nn.Linear(8 * 128, 256),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.base_head = nn.Linear(256, bases)
        self.tone_head = nn.Linear(256, 4)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling == "temporal8":
            features = self.frame_project(features).flatten(1)
        shared = self.project(features)
        return self.base_head(shared), self.tone_head(shared)


def score(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict, list[dict]]:
    model.eval()
    base_correct = tone_correct = joint_correct = tone_count = 0
    count = 0
    predictions = []
    with torch.inference_mode():
        for features, base, tone, rows in loader:
            base_logits, tone_logits = model(features.to(device))
            base_pred = base_logits.argmax(1).cpu()
            tone_pred = tone_logits.argmax(1).cpu()
            base_correct += (base_pred == base).sum().item()
            valid_tone = tone >= 0
            tone_correct += ((tone_pred == tone) & valid_tone).sum().item()
            joint_correct += (
                ((base_pred == base) & (tone_pred == tone) & valid_tone).sum().item()
            )
            tone_count += valid_tone.sum().item()
            count += base.numel()
            predictions.extend(
                {
                    "row": int(row),
                    "base_true": int(b),
                    "base_pred": int(bp),
                    "tone_true": int(t),
                    "tone_pred": int(tp),
                }
                for row, b, bp, t, tp in zip(rows, base, base_pred, tone, tone_pred)
            )
    metrics = {
        "n": count,
        "base_accuracy": base_correct / count if count else None,
        "tone_n": tone_count,
        "tone_accuracy": tone_correct / tone_count if tone_count else None,
        "joint_accuracy": joint_correct / tone_count if tone_count else None,
    }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = Path("configs/frozen_classifier.json")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--train-speaker", choices=["abe", "yue"], required=True)
    parser.add_argument("--pooling", choices=["global", "temporal8"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tone-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--validation-strategy",
        choices=["stratified-base", "hash-fraction"],
        default="stratified-base",
    )
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    apply_config_defaults(parser, requested_config_path(default_config))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    artifact = torch.load(args.features, map_location="cpu", weights_only=False)
    features = artifact[args.pooling]
    groups = [speaker_group(dataset) for dataset in artifact["datasets"]]
    base_names = sorted(set(artifact["bases"]))
    base_to_id = {name: index for index, name in enumerate(base_names)}
    base_targets = torch.tensor([base_to_id[name] for name in artifact["bases"]])
    # Neutral tone 5 and unspecified tones are masked from the symmetric 1--4 tone task.
    tone_targets = torch.tensor(
        [
            int(tone) - 1 if tone in {"1", "2", "3", "4"} else -1
            for tone in artifact["tones"]
        ]
    )

    train_indices, validation_indices = split_training_speaker(
        artifact["paths"],
        artifact["bases"],
        groups,
        args.train_speaker,
        args.validation_strategy,
        args.validation_fraction,
        args.seed,
    )
    external_indices, oli_indices = [], []
    external_speaker = "yue" if args.train_speaker == "abe" else "abe"
    for index, group in enumerate(groups):
        if group == external_speaker:
            external_indices.append(index)
        elif group == "oli":
            oli_indices.append(index)

    datasets = {
        "train": FeatureDataset(features, base_targets, tone_targets, train_indices),
        "validation": FeatureDataset(
            features, base_targets, tone_targets, validation_indices
        ),
        "external": FeatureDataset(
            features, base_targets, tone_targets, external_indices
        ),
        "oli": FeatureDataset(features, base_targets, tone_targets, oli_indices),
    }
    loaders = {
        name: DataLoader(
            data, batch_size=args.batch_size, shuffle=name == "train", num_workers=0
        )
        for name, data in datasets.items()
    }
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = Classifier(
        tuple(features.shape[1:]), args.pooling, len(base_names), args.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    base_loss_fn = nn.CrossEntropyLoss()
    tone_loss_fn = nn.CrossEntropyLoss()

    best_validation = -1.0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_features, base, tone, _ in loaders["train"]:
            batch_features, base, tone = (
                batch_features.to(device),
                base.to(device),
                tone.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            base_logits, tone_logits = model(batch_features)
            loss = base_loss_fn(base_logits, base)
            valid_tone = tone >= 0
            if valid_tone.any():
                loss = loss + args.tone_loss_weight * tone_loss_fn(
                    tone_logits[valid_tone], tone[valid_tone]
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * base.numel()
        validation, _ = score(model, loaders["validation"], device)
        selection = validation["base_accuracy"] + (validation["tone_accuracy"] or 0.0)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(datasets["train"]),
            **validation,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if selection > best_validation:
            best_validation = selection
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_revision = artifact.get("model_revision") or MODEL_REVISIONS.get(
        artifact["model_name"]
    )
    if model_revision is None:
        raise ValueError(f"No pinned revision is known for {artifact['model_name']!r}")
    metrics = {
        "checkpoint_kind": "frozen_encoder_classifier",
        "state_scope": "complete_head",
        "encoder": artifact["encoder"],
        "model_name": artifact["model_name"],
        "model_revision": model_revision,
        "train_speaker": args.train_speaker,
        "external_speaker": external_speaker,
        "pooling": args.pooling,
        "seed": args.seed,
        "validation_strategy": args.validation_strategy,
        "base_vocabulary_size": len(base_names),
        "base_vocabulary": base_names,
        "architecture": {
            "dropout": args.dropout,
            "projection_size": 256,
            "temporal_bins": 8 if args.pooling == "temporal8" else None,
            "frame_projection_size": 128 if args.pooling == "temporal8" else None,
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "tone_loss_weight": args.tone_loss_weight,
            "validation_fraction": args.validation_fraction,
        },
        "split_sizes": {name: len(data) for name, data in datasets.items()},
        "history": history,
    }
    all_predictions = {}
    for name in ("validation", "external", "oli"):
        metrics[name], all_predictions[name] = score(model, loaders[name], device)
    # Tone metrics for Oli are intentionally null because its labels are unspecified.
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata, measured_metrics = split_run_record(metrics)
    checkpoint = create_checkpoint(
        state_dict=best_state,
        metadata=metadata,
        metrics=measured_metrics,
    )
    save_checkpoint(args.output_dir / "classifier.pt", checkpoint)
    with (args.output_dir / "predictions.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ["split", "path", "base_true", "base_pred", "tone_true", "tone_pred"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for split, predictions in all_predictions.items():
            for prediction in predictions:
                row = prediction.pop("row")
                tone_true = (
                    prediction["tone_true"] + 1 if prediction["tone_true"] >= 0 else ""
                )
                writer.writerow(
                    {
                        "split": split,
                        "path": artifact["paths"][row],
                        "base_true": base_names[prediction["base_true"]],
                        "base_pred": base_names[prediction["base_pred"]],
                        "tone_true": tone_true,
                        "tone_pred": prediction["tone_pred"] + 1,
                    }
                )
    print(
        json.dumps(
            {key: metrics[key] for key in ("validation", "external", "oli")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
