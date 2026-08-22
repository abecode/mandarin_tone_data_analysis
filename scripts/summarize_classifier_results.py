#!/usr/bin/env python3
"""Export validation/test errors and confusion matrices for classifier runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def accuracy_errors(metrics: dict) -> dict[str, float | int | str]:
    return {
        "n": metrics["n"],
        "base_accuracy": metrics["base_accuracy"],
        "base_error": 1.0 - metrics["base_accuracy"],
        "tone_n": metrics["tone_n"],
        "tone_accuracy": metrics["tone_accuracy"]
        if metrics["tone_accuracy"] is not None
        else "",
        "tone_error": 1.0 - metrics["tone_accuracy"]
        if metrics["tone_accuracy"] is not None
        else "",
        "joint_accuracy": metrics["joint_accuracy"]
        if metrics["joint_accuracy"] is not None
        else "",
        "joint_error": 1.0 - metrics["joint_accuracy"]
        if metrics["joint_accuracy"] is not None
        else "",
    }


def write_matrix(path: Path, labels: list[str], counts: Counter) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["true\\pred", *labels])
        for true in labels:
            writer.writerow([true, *(counts[true, predicted] for predicted in labels)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "results_root",
        type=Path,
        nargs="?",
        default=Path("results/frozen_grid_stratified"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/frozen_grid_stratified_analysis")
    )
    parser.add_argument("--top-errors", type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    summary = []
    top_errors = []
    for metrics_path in sorted(args.results_root.glob("*/metrics.json")):
        run = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text())
        with (metrics_path.parent / "predictions.tsv").open(
            newline="", encoding="utf-8"
        ) as handle:
            predictions = list(csv.DictReader(handle, delimiter="\t"))
        for split in ("validation", "external"):
            row = {"run": run, "split": split, **accuracy_errors(metrics[split])}
            summary.append(row)
            selected = [item for item in predictions if item["split"] == split]

            base_counts = Counter(
                (item["base_true"], item["base_pred"]) for item in selected
            )
            write_matrix(
                args.output / f"{run}_{split}_base_confusion.tsv",
                metrics["base_vocabulary"],
                base_counts,
            )
            errors = Counter(
                (item["base_true"], item["base_pred"])
                for item in selected
                if item["base_true"] != item["base_pred"]
            )
            for rank, ((true, predicted), count) in enumerate(
                errors.most_common(args.top_errors), 1
            ):
                top_errors.append(
                    {
                        "run": run,
                        "split": split,
                        "rank": rank,
                        "base_true": true,
                        "base_pred": predicted,
                        "count": count,
                    }
                )

            tone_selected = [item for item in selected if item["tone_true"]]
            tone_counts = Counter(
                (item["tone_true"], item["tone_pred"]) for item in tone_selected
            )
            write_matrix(
                args.output / f"{run}_{split}_tone_confusion.tsv",
                ["1", "2", "3", "4"],
                tone_counts,
            )

    for filename, rows in (
        ("error_summary.tsv", summary),
        ("top_base_confusions.tsv", top_errors),
    ):
        with (args.output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=rows[0], delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"Wrote {len(summary)} split summaries and confusion matrices to {args.output}"
    )


if __name__ == "__main__":
    main()
