#!/usr/bin/env python3
"""Generate the student Colab notebook for ordered-attention tone training."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

OUTPUT = Path("notebooks/ordered_attention_tone_colab.ipynb")


def markdown(source: str) -> dict[str, object]:
    """Return a notebook Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    """Return a notebook code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip().splitlines(keepends=True),
    }


def main() -> None:
    """Write the notebook as deterministic notebook-format JSON."""
    cells = [
        markdown(
            """
            # Mandarin tone recognition with ordered temporal attention

            In this experiment you will train a four-tone classifier from the
            private `abecode/mandarin_isolated_syllables` dataset. A frozen
            Chinese HuBERT encoder converts each waveform into a sequence of
            acoustic representations. The model you train uses eight learned,
            ordered attention heads to summarize that sequence.

            We compare two otherwise identical training conditions:

            1. Original recordings only.
            2. Recordings with randomized leading and trailing silence.

            Speaker 1 supplies training and validation data. Speaker 2 is held
            out as the cross-speaker test set. Speaker 3 is not scored because
            its recordings do not have intended tone labels.

            **Important:** prompt labels are intended targets, not independent
            judgments of the tone actually produced. A model prediction is not
            a pronunciation-quality score.
            """
        ),
        markdown(
            """
            ## 1. Start a GPU runtime and install dependencies

            In Colab choose **Runtime → Change runtime type → T4 GPU** before
            continuing. The installation cell may ask you to restart the
            runtime if Colab already imported an incompatible package.
            """
        ),
        code(
            """
            %pip install -q -U \
                "huggingface_hub>=1.0,<2" \
                "transformers>=4.53,<6" \
                "gradio>=5,<7" \
                soundfile scipy scikit-learn pandas matplotlib
            """
        ),
        code(
            """
            import copy
            import getpass
            import math
            import random
            import zipfile
            from collections import defaultdict
            from pathlib import Path

            import gradio as gr
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import soundfile as sf
            import torch
            import torch.nn.functional as F
            from huggingface_hub import hf_hub_download
            from scipy.signal import resample_poly
            from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
            from sklearn.model_selection import train_test_split
            from torch import nn
            from torch.utils.data import DataLoader, Dataset
            from transformers import AutoFeatureExtractor, AutoModel

            SEED = 20260821
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            random.seed(SEED)
            print("Device:", DEVICE)
            if DEVICE.type != "cuda":
                print("Warning: feature extraction will be slow without a GPU.")
            """
        ),
        markdown(
            """
            ## 2. Authenticate and download the private dataset

            Add a read token to **Colab → Secrets** under the name `HF_TOKEN`
            and enable notebook access. Each student should use their own token.
            The fallback prompt hides the token but does not persist it.
            """
        ),
        code(
            """
            REPO_ID = "abecode/mandarin_isolated_syllables"
            REVISION = "v0.1.1"  # Use "main" before this tag exists.
            LOCAL_DATASET = Path("/content/mandarin_isolated_syllables")
            ARCHIVE_NAME = "mandarin_isolated_syllables_v0.1.1.zip"

            try:
                from google.colab import userdata
                HF_TOKEN = userdata.get("HF_TOKEN")
            except (ImportError, KeyError, RuntimeError):
                HF_TOKEN = None
            if not HF_TOKEN:
                HF_TOKEN = getpass.getpass("Hugging Face read token: ")

            archive_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=ARCHIVE_NAME,
                revision=REVISION,
                token=HF_TOKEN,
            )
            LOCAL_DATASET.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path) as archive:
                root = LOCAL_DATASET.resolve()
                for member in archive.infolist():
                    destination = (root / member.filename).resolve()
                    if not destination.is_relative_to(root):
                        raise ValueError(f"Unsafe archive path: {member.filename}")
                archive.extractall(LOCAL_DATASET)

            metadata = pd.read_csv(LOCAL_DATASET / "data" / "metadata.csv")
            metadata["audio_path"] = metadata["file_name"].map(
                lambda name: str(LOCAL_DATASET / "data" / name)
            )
            labeled = metadata[metadata["tone"].isin([1, 2, 3, 4])].copy()
            labeled["tone_index"] = labeled["tone"].astype(int) - 1
            print(metadata.groupby("speaker_id").size())
            print("Four-tone examples:", len(labeled))
            """
        ),
        markdown(
            """
            ## 3. Construct a reproducible cross-speaker experiment

            Validation is stratified by tone so each tone has approximately the
            same representation in training and validation. The primary test is
            stronger: every speaker-2 recording is excluded from training.

            The notebook uses the complete dataset by default. Set
            `CLASSROOM_MODE=True` only when demonstrating on a CPU runtime or
            when a shorter fallback run is needed.
            """
        ),
        code(
            """
            CLASSROOM_MODE = False
            MAX_TRAIN = 1600
            MAX_VALIDATION = 400
            MAX_TEST = 800
            AUGMENTED_VIEWS = 1  # Increase to 3 for more augmentation variety.
            EPOCHS = 10
            PATIENCE = 3

            speaker_1 = labeled[
                labeled["speaker_id"] == "speaker_000000001"
            ].copy()
            test_rows = labeled[
                labeled["speaker_id"] == "speaker_000000002"
            ].copy()
            train_rows, validation_rows = train_test_split(
                speaker_1,
                test_size=0.15,
                random_state=SEED,
                stratify=speaker_1["tone_index"],
            )

            def stratified_limit(frame, maximum, seed):
                if maximum is None or len(frame) <= maximum:
                    return frame.reset_index(drop=True)
                result, _ = train_test_split(
                    frame,
                    train_size=maximum,
                    random_state=seed,
                    stratify=frame["tone_index"],
                )
                return result.reset_index(drop=True)

            if CLASSROOM_MODE:
                train_rows = stratified_limit(train_rows, MAX_TRAIN, SEED)
                validation_rows = stratified_limit(
                    validation_rows, MAX_VALIDATION, SEED + 1
                )
                test_rows = stratified_limit(test_rows, MAX_TEST, SEED + 2)
            else:
                train_rows = train_rows.reset_index(drop=True)
                validation_rows = validation_rows.reset_index(drop=True)
                test_rows = test_rows.reset_index(drop=True)

            for name, frame in {
                "train": train_rows,
                "validation": validation_rows,
                "cross-speaker test": test_rows,
            }.items():
                print(name, len(frame), frame["tone"].value_counts().sort_index().to_dict())
            """
        ),
        markdown(
            """
            ## 4. Extract and cache frozen HuBERT frame representations

            This is the slow stage. HuBERT is frozen, so each waveform only
            needs to pass through it once. Cached sequences stay in CPU memory
            as float16 tensors; the small attention models train from them.

            The augmented view adds 0–500 ms of randomized low-level boundary
            material. This is a classroom approximation of the original study,
            which regenerated a richer mixture of endpoint nonspeech, matched
            noise, and zeros every epoch.
            """
        ),
        code(
            """
            MODEL_ID = "TencentGameMate/chinese-hubert-base"
            MODEL_REVISION = "fce0375452b1dd6c080ac3248d423d4d037bc831"
            TARGET_SAMPLE_RATE = 16_000
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                MODEL_ID, revision=MODEL_REVISION
            )
            encoder = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            encoder.eval().to(DEVICE)
            encoder.requires_grad_(False)
            HIDDEN_SIZE = encoder.config.hidden_size

            def load_waveform(path):
                waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
                waveform = waveform.mean(axis=1)
                if sample_rate != TARGET_SAMPLE_RATE:
                    divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
                    waveform = resample_poly(
                        waveform,
                        TARGET_SAMPLE_RATE // divisor,
                        sample_rate // divisor,
                    ).astype(np.float32)
                return waveform

            def add_boundary_silence(waveform, rng, maximum_ms=500):
                maximum = TARGET_SAMPLE_RATE * maximum_ms // 1000
                before = int(rng.integers(0, maximum + 1))
                after = int(rng.integers(0, maximum + 1))
                rms = float(np.sqrt(np.mean(np.square(waveform)) + 1e-9))

                def boundary(length):
                    if rng.random() < 0.5:
                        return np.zeros(length, dtype=np.float32)
                    return rng.normal(0, max(rms * 0.01, 1e-5), length).astype(np.float32)

                return np.concatenate((boundary(before), waveform, boundary(after)))

            def extract_batch(waveforms):
                inputs = feature_extractor(
                    waveforms,
                    sampling_rate=TARGET_SAMPLE_RATE,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                values = inputs.input_values.to(DEVICE)
                mask = inputs.attention_mask.to(DEVICE)
                with torch.inference_mode(), torch.autocast(
                    device_type=DEVICE.type, enabled=DEVICE.type == "cuda"
                ):
                    hidden = encoder(
                        input_values=values,
                        attention_mask=mask,
                    ).last_hidden_state
                lengths = encoder._get_feat_extract_output_lengths(mask.sum(1))
                return [
                    sequence[: int(length)].detach().cpu().to(torch.float16)
                    for sequence, length in zip(hidden, lengths)
                ]

            def extract_frame(frame, augmented=False, view=0, batch_size=8):
                features = []
                rng = np.random.default_rng(SEED + 10_000 * view)
                for start in range(0, len(frame), batch_size):
                    batch = frame.iloc[start : start + batch_size]
                    waveforms = [load_waveform(path) for path in batch["audio_path"]]
                    if augmented:
                        waveforms = [add_boundary_silence(waveform, rng) for waveform in waveforms]
                    features.extend(extract_batch(waveforms))
                    if len(features) % 200 < batch_size:
                        print(f"Extracted {len(features)}/{len(frame)}")
                return features

            original_features = {
                "train": extract_frame(train_rows),
                "validation": extract_frame(validation_rows),
                "test": extract_frame(test_rows),
            }
            augmented_train_features = [
                extract_frame(train_rows, augmented=True, view=view + 1)
                for view in range(AUGMENTED_VIEWS)
            ]
            encoder.to("cpu")
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            print("Feature cache complete")
            """
        ),
        markdown(
            """
            ## 5. Define the ordered eight-head tone classifier

            Each attention head combines acoustic content with a learnable
            Gaussian preference for a relative time region. Ordering loss
            discourages adjacent heads from crossing. Diversity loss discourages
            all heads from learning the same weighting pattern.
            """
        ),
        code(
            """
            def sequence_mask(lengths, width):
                positions = torch.arange(width, device=lengths.device).unsqueeze(0)
                return positions < lengths.unsqueeze(1)

            class OrderedAttentionPooling(nn.Module):
                def __init__(self, width, heads=8, projection_size=128, dropout=0.2):
                    super().__init__()
                    self.heads = heads
                    attention_size = 128
                    self.keys = nn.Sequential(nn.Linear(width, attention_size), nn.Tanh())
                    self.queries = nn.Parameter(torch.empty(heads, attention_size))
                    nn.init.normal_(self.queries, std=0.02)
                    centers = torch.linspace(0.08, 0.92, heads)
                    self.center_logits = nn.Parameter(torch.logit(centers))
                    self.log_widths = nn.Parameter(
                        torch.full((heads,), math.log(0.22))
                    )
                    self.frame_project = nn.Sequential(
                        nn.LayerNorm(width),
                        nn.Linear(width, projection_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )

                def forward(self, hidden, lengths):
                    batch, frames, _ = hidden.shape
                    valid = sequence_mask(lengths, frames)
                    keys = self.keys(hidden)
                    content = torch.einsum(
                        "btd,hd->bht", keys, self.queries
                    ) / math.sqrt(keys.shape[-1])
                    positions = torch.arange(
                        frames, device=hidden.device
                    ).view(1, 1, -1)
                    denominators = (lengths - 1).clamp_min(1).view(-1, 1, 1)
                    relative_positions = positions / denominators
                    centers = self.center_logits.sigmoid().view(1, -1, 1)
                    widths = self.log_widths.exp().clamp(0.05, 1.0).view(1, -1, 1)
                    position_bias = -0.5 * (
                        (relative_positions - centers) / widths
                    ).square()
                    logits = (content + position_bias).masked_fill(
                        ~valid.unsqueeze(1), -torch.inf
                    )
                    attention = logits.softmax(dim=2)
                    projected = self.frame_project(hidden)
                    summaries = torch.einsum(
                        "bht,btd->bhd", attention, projected
                    ).flatten(1)
                    empirical_centers = (attention * relative_positions).sum(dim=2)
                    normalized = attention / attention.square().sum(
                        dim=2, keepdim=True
                    ).sqrt()
                    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
                    identity = torch.eye(self.heads, device=hidden.device).unsqueeze(0)
                    diversity_loss = (similarity - identity).square().mean()
                    ordering_loss = torch.relu(
                        empirical_centers[:, :-1]
                        - empirical_centers[:, 1:]
                        + 0.03
                    ).mean()
                    return summaries, {
                        "attention": attention,
                        "centers": empirical_centers,
                        "diversity_loss": diversity_loss,
                        "ordering_loss": ordering_loss,
                    }

            class ToneModel(nn.Module):
                def __init__(self, hidden_size, dropout=0.2):
                    super().__init__()
                    self.pooling = OrderedAttentionPooling(hidden_size, dropout=dropout)
                    self.project = nn.Sequential(
                        nn.LayerNorm(8 * 128),
                        nn.Linear(8 * 128, 256),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    self.tone_head = nn.Linear(256, 4)

                def forward(self, hidden, lengths):
                    summary, auxiliary = self.pooling(hidden, lengths)
                    return self.tone_head(self.project(summary)), auxiliary
            """
        ),
        code(
            """
            class CachedFeatureDataset(Dataset):
                def __init__(self, frame, original, augmented=None):
                    self.labels = torch.tensor(
                        frame["tone_index"].to_numpy(), dtype=torch.long
                    )
                    self.original = original
                    self.augmented = augmented

                def __len__(self):
                    return len(self.labels)

                def __getitem__(self, index):
                    if self.augmented:
                        view = random.randrange(len(self.augmented))
                        feature = self.augmented[view][index]
                    else:
                        feature = self.original[index]
                    return feature, self.labels[index]

            def collate_features(items):
                features, labels = zip(*items)
                lengths = torch.tensor([len(feature) for feature in features])
                padded = nn.utils.rnn.pad_sequence(
                    [feature.float() for feature in features], batch_first=True
                )
                return padded, lengths, torch.stack(labels)

            def make_loader(frame, features, augmented=None, shuffle=False):
                return DataLoader(
                    CachedFeatureDataset(frame, features, augmented),
                    batch_size=64,
                    shuffle=shuffle,
                    collate_fn=collate_features,
                    num_workers=0,
                    pin_memory=DEVICE.type == "cuda",
                )

            validation_loader = make_loader(
                validation_rows, original_features["validation"]
            )
            test_loader = make_loader(test_rows, original_features["test"])
            """
        ),
        markdown(
            """
            ## 6. Train the paired experiment

            Both conditions start from the same model initialization. We select
            the epoch with the highest validation tone accuracy and never use
            speaker-2 test labels for model selection.
            """
        ),
        code(
            """
            @torch.inference_mode()
            def evaluate(model, loader):
                model.eval()
                predictions, targets = [], []
                for hidden, lengths, labels in loader:
                    logits, _ = model(hidden.to(DEVICE), lengths.to(DEVICE))
                    predictions.extend(logits.argmax(1).cpu().tolist())
                    targets.extend(labels.tolist())
                return accuracy_score(targets, predictions), targets, predictions

            def train_condition(name, augmented):
                torch.manual_seed(SEED)
                model = ToneModel(HIDDEN_SIZE).to(DEVICE)
                optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
                loader = make_loader(
                    train_rows,
                    original_features["train"],
                    augmented=augmented_train_features if augmented else None,
                    shuffle=True,
                )
                best_accuracy = -1.0
                best_state = None
                stale_epochs = 0
                history = defaultdict(list)
                for epoch in range(1, EPOCHS + 1):
                    model.train()
                    losses = []
                    for hidden, lengths, labels in loader:
                        hidden = hidden.to(DEVICE, non_blocking=True)
                        lengths = lengths.to(DEVICE, non_blocking=True)
                        labels = labels.to(DEVICE, non_blocking=True)
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
                    validation_accuracy, _, _ = evaluate(model, validation_loader)
                    history["loss"].append(float(np.mean(losses)))
                    history["validation_accuracy"].append(validation_accuracy)
                    print(
                        f"{name}: epoch {epoch:02d}, "
                        f"loss={np.mean(losses):.4f}, "
                        f"validation={validation_accuracy:.3f}"
                    )
                    if validation_accuracy > best_accuracy:
                        best_accuracy = validation_accuracy
                        best_state = copy.deepcopy(model.state_dict())
                        stale_epochs = 0
                    else:
                        stale_epochs += 1
                        if stale_epochs >= PATIENCE:
                            print("Early stopping")
                            break
                model.load_state_dict(best_state)
                return model, dict(history)

            original_model, original_history = train_condition(
                "original", augmented=False
            )
            augmented_model, augmented_history = train_condition(
                "augmented", augmented=True
            )
            """
        ),
        markdown(
            """
            ## 7. Compare validation and cross-speaker performance

            The confusion matrices put intended tones on rows and predicted
            tones on columns. Compare validation with the held-out speaker test:
            high validation performance does not guarantee generalization to a
            different voice or recording setup.
            """
        ),
        code(
            """
            results = {}
            for name, model in {
                "Original": original_model,
                "Silence augmented": augmented_model,
            }.items():
                validation = evaluate(model, validation_loader)
                test = evaluate(model, test_loader)
                results[name] = {"validation": validation, "test": test}
                print(
                    f"{name:18s} validation={validation[0]:.3f} "
                    f"cross-speaker test={test[0]:.3f}"
                )

            figure, axes = plt.subplots(2, 2, figsize=(10, 9))
            for row, (name, conditions) in enumerate(results.items()):
                for column, split in enumerate(("validation", "test")):
                    _, targets, predictions = conditions[split]
                    ConfusionMatrixDisplay.from_predictions(
                        np.asarray(targets) + 1,
                        np.asarray(predictions) + 1,
                        labels=[1, 2, 3, 4],
                        normalize="true",
                        values_format=".2f",
                        ax=axes[row, column],
                        colorbar=False,
                    )
                    axes[row, column].set_title(f"{name}: {split}")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 8. Inspect the learned attention heads

            These are broad, overlapping temporal summaries—not hard phonetic
            boundaries. The horizontal axis is relative time, so recordings of
            different lengths are comparable.
            """
        ),
        code(
            """
            @torch.inference_mode()
            def attention_figure(model, feature, title):
                model.eval()
                hidden = feature.float().unsqueeze(0).to(DEVICE)
                lengths = torch.tensor([len(feature)], device=DEVICE)
                logits, auxiliary = model(hidden, lengths)
                probabilities = logits.softmax(1)[0].cpu().numpy()
                attention = auxiliary["attention"][0].cpu().numpy()
                positions = np.linspace(0, 1, attention.shape[1])
                figure, axis = plt.subplots(figsize=(10, 4))
                for head, weights in enumerate(attention, 1):
                    axis.plot(positions, weights, label=f"Head {head}")
                axis.set(xlabel="Relative time", ylabel="Attention weight", title=title)
                axis.legend(ncol=4, fontsize=8)
                figure.tight_layout()
                return figure, probabilities

            example_index = 0
            figure, probabilities = attention_figure(
                augmented_model,
                original_features["test"][example_index],
                "Ordered attention on one speaker-2 recording",
            )
            print({f"Tone {index + 1}": float(value) for index, value in enumerate(probabilities)})
            display(figure)
            """
        ),
        markdown(
            """
            ## 9. Try the trained model with a microphone or audio file

            Say one isolated Mandarin syllable. The demo reports tone
            probabilities from both newly trained conditions and visualizes the
            augmented model's attention. Recordings are processed in this Colab
            runtime and are not added to the dataset. Colab requires Gradio to
            create a temporary public share URL, so do not post that URL or use
            it for sensitive recordings.
            """
        ),
        code(
            """
            encoder.to(DEVICE).eval()

            def prepare_gradio_audio(audio):
                if audio is None:
                    raise gr.Error("Please record or upload an isolated syllable.")
                sample_rate, waveform = audio
                waveform = np.asarray(waveform)
                if waveform.ndim == 2:
                    waveform = waveform.mean(axis=1)
                if np.issubdtype(waveform.dtype, np.integer):
                    maximum = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
                    waveform = waveform.astype(np.float32) / maximum
                else:
                    waveform = waveform.astype(np.float32)
                if sample_rate != TARGET_SAMPLE_RATE:
                    divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
                    waveform = resample_poly(
                        waveform,
                        TARGET_SAMPLE_RATE // divisor,
                        int(sample_rate) // divisor,
                    ).astype(np.float32)
                if len(waveform) > 10 * TARGET_SAMPLE_RATE:
                    waveform = waveform[: 10 * TARGET_SAMPLE_RATE]
                return waveform

            def probability_labels(probabilities):
                return {
                    f"Tone {tone}": float(probabilities[tone - 1])
                    for tone in range(1, 5)
                }

            def demo_predict(audio):
                waveform = prepare_gradio_audio(audio)
                feature = extract_batch([waveform])[0]
                figure, augmented_probabilities = attention_figure(
                    augmented_model,
                    feature,
                    "Augmented model: eight ordered attention heads",
                )
                _, original_probabilities = attention_figure(
                    original_model,
                    feature,
                    "Original model",
                )
                return (
                    probability_labels(original_probabilities),
                    probability_labels(augmented_probabilities),
                    figure,
                )

            with gr.Blocks() as demo:
                gr.Markdown(
                    "# Try the tone classifiers\\n"
                    "Record one isolated Mandarin syllable. These predictions "
                    "are experimental and are not pronunciation assessments."
                )
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Isolated syllable",
                )
                run_button = gr.Button("Predict tone", variant="primary")
                with gr.Row():
                    original_output = gr.Label(label="No augmentation")
                    augmented_output = gr.Label(label="Silence augmentation")
                attention_output = gr.Plot(label="Ordered attention")
                run_button.click(
                    demo_predict,
                    inputs=audio_input,
                    outputs=[original_output, augmented_output, attention_output],
                )
            demo.launch(share=True, debug=False)
            """
        ),
        markdown(
            """
            ## Questions for discussion

            1. Which tone pairs are most often confused on validation and test?
            2. Does silence augmentation improve cross-speaker performance in
               your run? Is the effect the same for every tone?
            3. Do the eight heads remain ordered? Do they attend to distinct or
               overlapping regions?
            4. Try adding silence manually before speaking. Which model is more
               stable?
            5. Why should these predictions not be interpreted as pronunciation
               grades?
            """
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": OUTPUT.name, "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
