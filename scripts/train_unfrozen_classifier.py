#!/usr/bin/env python3
"""Partially fine-tune a speech encoder with syllable and tone heads."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from attention_pooling import (
    ATTENTION_POOLING,
    AttentiveStatisticsPooling,
    OrderedAttentionPooling,
)
from checkpoint_utils import (
    MODEL_REVISIONS,
    create_checkpoint,
    save_checkpoint,
    split_run_record,
)
from config_utils import apply_config_defaults, requested_config_path
from extract_speech_features import MODEL_NAMES
from silence_augmentation import (
    SilenceAugmentationConfig,
    add_boundary_silence,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_syllable_classifier import speaker_group, split_training_speaker
from transformers import AutoModel

TEMPORAL_POOLING = {
    "temporal8": {"bins": 8, "frame_projection_size": 128},
    "temporal16": {"bins": 16, "frame_projection_size": 64},
}


class AudioDataset(Dataset):
    def __init__(
        self, artifact: dict, base: torch.Tensor, tone: torch.Tensor, indices: list[int]
    ):
        self.artifact = artifact
        self.base = base
        self.tone = tone
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        row = self.indices[index]
        return self.artifact["waveforms"][row], self.base[row], self.tone[row], row


def collate_waveforms(items):
    length = max(item[0].numel() for item in items)
    values = torch.zeros(len(items), length)
    mask = torch.zeros(len(items), length, dtype=torch.long)
    base = torch.empty(len(items), dtype=torch.long)
    tone = torch.empty(len(items), dtype=torch.long)
    rows = torch.empty(len(items), dtype=torch.long)
    for index, (audio, base_target, tone_target, row) in enumerate(items):
        values[index, : audio.numel()] = audio
        mask[index, : audio.numel()] = 1
        base[index], tone[index], rows[index] = base_target, tone_target, row
    return values, mask, base, tone, rows


class AudioCollator:
    """Optionally add boundary nonspeech before padding an audio batch."""

    def __init__(
        self,
        augment: bool = False,
        endpoint_metadata: list[dict] | None = None,
        augmentation_config: SilenceAugmentationConfig = SilenceAugmentationConfig(),
    ) -> None:
        self.augment = augment
        self.endpoint_metadata = endpoint_metadata
        self.augmentation_config = augmentation_config

    def __call__(self, items):
        if not self.augment:
            return collate_waveforms(items)
        if self.endpoint_metadata is None:
            raise ValueError("Silence augmentation requires endpoint metadata")
        augmented = []
        for waveform, base, tone, row in items:
            waveform, _ = add_boundary_silence(
                waveform,
                self.endpoint_metadata[row],
                self.augmentation_config,
            )
            augmented.append((waveform, base, tone, row))
        return collate_waveforms(augmented)


class FineTuneModel(nn.Module):
    def __init__(self, encoder: nn.Module, pooling: str, bases: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        width = encoder.config.hidden_size
        self.attentive_global = None
        self.ordered_attention = None
        if pooling == "global":
            self.temporal_bins = None
            self.frame_projection_size = None
            self.project = nn.Sequential(
                nn.LayerNorm(width * 2),
                nn.Linear(width * 2, 256),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif pooling in TEMPORAL_POOLING:
            temporal_config = TEMPORAL_POOLING[pooling]
            self.temporal_bins = temporal_config["bins"]
            self.frame_projection_size = temporal_config["frame_projection_size"]
            self.frame_project = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, self.frame_projection_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.project = nn.Sequential(
                nn.LayerNorm(self.temporal_bins * self.frame_projection_size),
                nn.Linear(self.temporal_bins * self.frame_projection_size, 256),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif pooling == "attentive_global":
            self.temporal_bins = None
            self.frame_projection_size = None
            self.attentive_global = AttentiveStatisticsPooling(width)
            self.project = self._projector(width * 2, 256, dropout)
        elif pooling == "ordered8":
            self.temporal_bins = 8
            self.frame_projection_size = 128
            self.ordered_attention = OrderedAttentionPooling(width, dropout=dropout)
            self.project = self._projector(8 * 128, 256, dropout)
        elif pooling == "attentive_combined":
            self.temporal_bins = 8
            self.frame_projection_size = 128
            self.attentive_global = AttentiveStatisticsPooling(width)
            self.ordered_attention = OrderedAttentionPooling(width, dropout=dropout)
            self.global_project = self._projector(width * 2, 128, dropout)
            self.ordered_project = self._projector(8 * 128, 128, dropout)
            self.project = self._projector(256, 256, dropout)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.base_head = nn.Linear(256, bases)
        self.tone_head = nn.Linear(256, 4)

    @staticmethod
    def _projector(input_size: int, output_size: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def aggregate(
        self, hidden: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.pooling in TEMPORAL_POOLING:
            pooled = []
            for sequence, length in zip(hidden, lengths.tolist()):
                sequence = self.frame_project(sequence[: max(1, int(length))])
                pooled.append(
                    F.adaptive_avg_pool1d(
                        sequence.T.unsqueeze(0), self.temporal_bins
                    ).flatten()
                )
            return torch.stack(pooled), {}
        if self.pooling in ATTENTION_POOLING:
            auxiliary = {}
            if self.attentive_global is not None:
                global_summary, global_attention = self.attentive_global(
                    hidden, lengths
                )
                auxiliary["global_attention"] = global_attention
            if self.ordered_attention is not None:
                ordered_summary, ordered_auxiliary = self.ordered_attention(
                    hidden, lengths
                )
                auxiliary.update(ordered_auxiliary)
            if self.pooling == "attentive_global":
                return global_summary, auxiliary
            if self.pooling == "ordered8":
                return ordered_summary, auxiliary
            combined = torch.cat(
                (
                    self.global_project(global_summary),
                    self.ordered_project(ordered_summary),
                ),
                dim=1,
            )
            return combined, auxiliary
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = positions < lengths.unsqueeze(1)
        weight = mask.unsqueeze(-1).to(hidden.dtype)
        denominator = lengths.clamp_min(1).to(hidden.dtype).view(-1, 1)
        mean = (hidden * weight).sum(1) / denominator
        variance = ((hidden - mean.unsqueeze(1)).square() * weight).sum(1) / denominator
        return torch.cat((mean, variance.clamp_min(1e-7).sqrt()), dim=1), {}

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor, return_auxiliary: bool = False
    ):
        hidden = self.encoder(
            input_values=values, attention_mask=mask
        ).last_hidden_state
        lengths = self.encoder._get_feat_extract_output_lengths(mask.sum(1)).to(
            hidden.device
        )
        aggregate, auxiliary = self.aggregate(hidden, lengths)
        shared = self.project(aggregate)
        output = self.base_head(shared), self.tone_head(shared)
        if return_auxiliary:
            return *output, auxiliary
        return output


def unfreeze_top_layers(encoder: nn.Module, count: int) -> list[str]:
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    layers = encoder.encoder.layers
    if not 1 <= count <= len(layers):
        raise ValueError(f"--unfreeze-layers must be from 1 to {len(layers)}")
    for layer in layers[-count:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    return [
        f"encoder.layers.{index}" for index in range(len(layers) - count, len(layers))
    ]


def evaluate(model, loader, device, base_names, paths) -> tuple[dict, list[dict]]:
    model.eval()
    base_correct = tone_correct = joint_correct = tone_count = count = 0
    predictions = []
    attention_centers = []
    attention_entropies = []
    with torch.inference_mode():
        for values, mask, base, tone, rows in loader:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                base_logits, tone_logits, auxiliary = model(
                    values.to(device), mask.to(device), return_auxiliary=True
                )
            base_pred, tone_pred = (
                base_logits.argmax(1).cpu(),
                tone_logits.argmax(1).cpu(),
            )
            valid = tone >= 0
            base_correct += (base_pred == base).sum().item()
            tone_correct += ((tone_pred == tone) & valid).sum().item()
            joint_correct += (
                ((base_pred == base) & (tone_pred == tone) & valid).sum().item()
            )
            count += base.numel()
            tone_count += valid.sum().item()
            centers = auxiliary.get("centers")
            entropies = auxiliary.get("entropy")
            if centers is not None:
                center_rows = centers.float().cpu().tolist()
                entropy_rows = entropies.float().cpu().tolist()
                attention_centers.extend(center_rows)
                attention_entropies.extend(entropy_rows)
            else:
                center_rows = [None] * rows.numel()
            predictions.extend(
                {
                    "path": paths[int(row)],
                    "base_true": base_names[int(b)],
                    "base_pred": base_names[int(bp)],
                    "tone_true": int(t) + 1 if int(t) >= 0 else "",
                    "tone_pred": int(tp) + 1,
                    "attention_centers": (
                        ";".join(f"{value:.5f}" for value in center_row)
                        if center_row is not None
                        else ""
                    ),
                }
                for row, b, bp, t, tp, center_row in zip(
                    rows, base, base_pred, tone, tone_pred, center_rows
                )
            )
    metrics = {
        "n": count,
        "base_accuracy": base_correct / count if count else None,
        "tone_n": tone_count,
        "tone_accuracy": tone_correct / tone_count if tone_count else None,
        "joint_accuracy": joint_correct / tone_count if tone_count else None,
    }
    if attention_centers:
        center_tensor = torch.tensor(attention_centers)
        entropy_tensor = torch.tensor(attention_entropies)
        metrics["attention_mean_centers"] = center_tensor.mean(0).tolist()
        metrics["attention_mean_entropy"] = entropy_tensor.mean(0).tolist()
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    default_config = Path("configs/unfrozen_classifier.json")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--audio-cache", type=Path, default=Path("data/audio_16khz.pt"))
    parser.add_argument(
        "--endpoint-cache", type=Path, default=Path("data/audio_16khz_endpointed.pt")
    )
    parser.add_argument("--encoder", choices=sorted(MODEL_NAMES), required=True)
    parser.add_argument("--model-cache", type=Path, default=Path("models/huggingface"))
    parser.add_argument("--train-speaker", choices=["abe", "yue"], required=True)
    parser.add_argument(
        "--pooling",
        choices=["global", *TEMPORAL_POOLING, *sorted(ATTENTION_POOLING)],
        required=True,
    )
    parser.add_argument("--silence-augmentation", action="store_true")
    parser.add_argument("--maximum-silence-ms", type=int, default=500)
    parser.add_argument("--attention-diversity-weight", type=float, default=0.01)
    parser.add_argument("--attention-ordering-weight", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--tone-loss-weight", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="cuda")
    apply_config_defaults(parser, requested_config_path(default_config))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    artifact = torch.load(args.audio_cache, map_location="cpu", weights_only=False)
    endpoint_metadata = None
    if args.silence_augmentation:
        endpoint_artifact = torch.load(
            args.endpoint_cache, map_location="cpu", weights_only=False
        )
        if endpoint_artifact["paths"] != artifact["paths"]:
            raise ValueError("Audio and endpoint caches contain different path orders")
        endpoint_metadata = endpoint_artifact["endpoint_metadata"]
        if endpoint_metadata is None:
            raise ValueError("Endpoint cache does not contain endpoint metadata")
    groups = [speaker_group(dataset) for dataset in artifact["datasets"]]
    base_names = sorted(set(artifact["bases"]))
    base_to_id = {name: i for i, name in enumerate(base_names)}
    base_targets = torch.tensor([base_to_id[name] for name in artifact["bases"]])
    tone_targets = torch.tensor(
        [
            int(tone) - 1 if tone in {"1", "2", "3", "4"} else -1
            for tone in artifact["tones"]
        ]
    )
    train, validation = split_training_speaker(
        artifact["paths"],
        artifact["bases"],
        groups,
        args.train_speaker,
        "stratified-base",
        0.15,
        args.seed,
    )
    external_speaker = "yue" if args.train_speaker == "abe" else "abe"
    external = [i for i, group in enumerate(groups) if group == external_speaker]
    oli = [i for i, group in enumerate(groups) if group == "oli"]
    indices = {
        "train": train,
        "validation": validation,
        "external": external,
        "oli": oli,
    }
    datasets = {
        name: AudioDataset(artifact, base_targets, tone_targets, rows)
        for name, rows in indices.items()
    }
    augmentation_config = SilenceAugmentationConfig(maximum_ms=args.maximum_silence_ms)
    loaders = {
        name: DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=name == "train",
            collate_fn=AudioCollator(
                augment=name == "train" and args.silence_augmentation,
                endpoint_metadata=endpoint_metadata,
                augmentation_config=augmentation_config,
            ),
            num_workers=0,
            pin_memory=True,
        )
        for name, data in datasets.items()
    }

    encoder = AutoModel.from_pretrained(
        MODEL_NAMES[args.encoder],
        revision=MODEL_REVISIONS[MODEL_NAMES[args.encoder]],
        cache_dir=args.model_cache,
        local_files_only=True,
    )
    unfrozen = unfreeze_top_layers(encoder, args.unfreeze_layers)
    model = FineTuneModel(encoder, args.pooling, len(base_names), args.dropout).to(
        device
    )
    encoder_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    head_parameters = [
        p for name, p in model.named_parameters() if not name.startswith("encoder.")
    ]
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": args.encoder_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ],
        weight_decay=1e-2,
    )
    base_loss_fn, tone_loss_fn = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()

    best_scores = {
        "base_then_tone": (-1.0, -1.0),
        "tone": -1.0,
        "joint": -1.0,
    }
    best_states = {}
    best_epochs = {}
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for step, (values, mask, base, tone, _) in enumerate(loaders["train"], 1):
            values, mask, base, tone = (
                values.to(device),
                mask.to(device),
                base.to(device),
                tone.to(device),
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                base_logits, tone_logits, auxiliary = model(
                    values, mask, return_auxiliary=True
                )
                loss = base_loss_fn(base_logits, base)
                valid = tone >= 0
                if valid.any():
                    loss = loss + args.tone_loss_weight * tone_loss_fn(
                        tone_logits[valid], tone[valid]
                    )
                if "diversity_loss" in auxiliary:
                    loss = (
                        loss
                        + args.attention_diversity_weight * auxiliary["diversity_loss"]
                        + args.attention_ordering_weight * auxiliary["ordering_loss"]
                    )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            total_loss += loss.item() * base.numel()
            if step % args.gradient_accumulation == 0 or step == len(loaders["train"]):
                nn.utils.clip_grad_norm_([*encoder_parameters, *head_parameters], 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_metrics, _ = evaluate(
            model, loaders["validation"], device, base_names, artifact["paths"]
        )
        selections = {
            "base_then_tone": (
                validation_metrics["base_accuracy"],
                validation_metrics["tone_accuracy"] or 0.0,
            ),
            "tone": validation_metrics["tone_accuracy"] or 0.0,
            "joint": validation_metrics["joint_accuracy"] or 0.0,
        }
        primary_selection = selections["base_then_tone"]
        record = {
            "epoch": epoch,
            "train_loss": total_loss / len(datasets["train"]),
            **validation_metrics,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        for criterion, selection in selections.items():
            if selection > best_scores[criterion]:
                best_scores[criterion] = selection
                best_epochs[criterion] = epoch
                best_states[criterion] = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                    if name in trainable_names
                }
        if primary_selection > best_scores["base_then_tone"]:
            raise AssertionError("Primary checkpoint tracking is inconsistent")
        if best_epochs["base_then_tone"] == epoch:
            stale = 0
        else:
            if epoch >= args.minimum_epochs:
                stale += 1
            if epoch >= args.minimum_epochs and stale >= args.patience:
                break

    best_state = best_states["base_then_tone"]
    model.load_state_dict(best_state, strict=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "checkpoint_kind": "partial_finetune",
        "state_scope": "trainable_overlay",
        "encoder": args.encoder,
        "model_name": MODEL_NAMES[args.encoder],
        "model_revision": MODEL_REVISIONS[MODEL_NAMES[args.encoder]],
        "train_speaker": args.train_speaker,
        "external_speaker": external_speaker,
        "pooling": args.pooling,
        "unfreeze_layers": args.unfreeze_layers,
        "unfrozen_components": unfrozen,
        "seed": args.seed,
        "validation_strategy": "stratified-base",
        "base_vocabulary_size": len(base_names),
        "base_vocabulary": base_names,
        "architecture": {
            "dropout": args.dropout,
            "projection_size": 256,
            "temporal_bins": (
                TEMPORAL_POOLING[args.pooling]["bins"]
                if args.pooling in TEMPORAL_POOLING
                else 8
                if args.pooling in {"ordered8", "attentive_combined"}
                else None
            ),
            "frame_projection_size": (
                TEMPORAL_POOLING[args.pooling]["frame_projection_size"]
                if args.pooling in TEMPORAL_POOLING
                else 128
                if args.pooling in {"ordered8", "attentive_combined"}
                else None
            ),
            "attention_heads": (
                8 if args.pooling in {"ordered8", "attentive_combined"} else None
            ),
            "attention_size": 128 if args.pooling in ATTENTION_POOLING else None,
        },
        "training": {
            "epochs": args.epochs,
            "patience": args.patience,
            "minimum_epochs": args.minimum_epochs,
            "checkpoint_selection": "base_accuracy_then_tone_accuracy",
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "tone_loss_weight": args.tone_loss_weight,
            "attention_diversity_weight": args.attention_diversity_weight,
            "attention_ordering_weight": args.attention_ordering_weight,
            "silence_augmentation": args.silence_augmentation,
            "silence_augmentation_config": (
                asdict(augmentation_config) if args.silence_augmentation else None
            ),
        },
        "split_sizes": {name: len(data) for name, data in datasets.items()},
        "history": history,
        "selected_epochs": best_epochs,
        "selected_epoch": best_epochs["base_then_tone"],
    }
    predictions = {}
    for name in ("validation", "external", "oli"):
        metrics[name], predictions[name] = evaluate(
            model, loaders[name], device, base_names, artifact["paths"]
        )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata, measured_metrics = split_run_record(metrics)
    checkpoint = create_checkpoint(
        state_dict=best_state,
        metadata=metadata,
        metrics=measured_metrics,
    )
    save_checkpoint(args.output_dir / "classifier.pt", checkpoint)
    for criterion, filename in (
        ("tone", "classifier_best_tone.pt"),
        ("joint", "classifier_best_joint.pt"),
    ):
        if best_epochs[criterion] == best_epochs["base_then_tone"]:
            alternate_metrics = measured_metrics
        else:
            model.load_state_dict(best_states[criterion], strict=False)
            alternate_metrics = {
                "history": history,
                **{
                    name: evaluate(
                        model, loaders[name], device, base_names, artifact["paths"]
                    )[0]
                    for name in ("validation", "external", "oli")
                },
            }
        alternate_metadata = copy.deepcopy(metadata)
        alternate_metadata["training"]["checkpoint_selection"] = criterion
        alternate_metadata["selected_epoch"] = best_epochs[criterion]
        alternate_checkpoint = create_checkpoint(
            state_dict=best_states[criterion],
            metadata=alternate_metadata,
            metrics=alternate_metrics,
        )
        save_checkpoint(args.output_dir / filename, alternate_checkpoint)
    with (args.output_dir / "predictions.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = [
            "split",
            "path",
            "base_true",
            "base_pred",
            "tone_true",
            "tone_pred",
            "attention_centers",
        ]
        writer = csv.DictWriter(
            handle, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for split, rows in predictions.items():
            for row in rows:
                writer.writerow({"split": split, **row})
    print(
        json.dumps(
            {name: metrics[name] for name in ("validation", "external", "oli")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
