#!/usr/bin/env python3
"""Evaluate an ensemble of saved SAR fine-tuning runs.

Each run directory must contain the `test_logits.pt` produced by
`train_sar_finetune.py`. By default this averages the TTA logits from three
ResNet18 seed runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved SAR TTA logits.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[
            Path("results/sar_finetune/resnet18_size160_seed42"),
            Path("results/sar_finetune/resnet18_size160_seed7"),
            Path("results/sar_finetune/resnet18_size160_seed123"),
        ],
        type=Path,
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=None,
        type=float,
        help="Optional run weights. Defaults to equal weights.",
    )
    parser.add_argument(
        "--logits-key",
        default="tta_logits",
        choices=["logits", "tta_logits"],
    )
    parser.add_argument(
        "--output",
        default=Path("results/sar_finetune/tta_ensemble_metrics.json"),
        type=Path,
    )
    return parser.parse_args()


def confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(label)][int(prediction)] += 1
    return matrix


def per_class_accuracy(matrix: list[list[int]], class_names: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        total = sum(matrix[index])
        result[class_name] = matrix[index][index] / total if total else 0.0
    return result


def main() -> None:
    args = parse_args()
    if args.weights is None:
        weights = [1.0 for _ in args.runs]
    else:
        weights = args.weights
    if len(weights) != len(args.runs):
        raise ValueError("--weights must match --runs length")
    if sum(weights) <= 0:
        raise ValueError("--weights must sum to a positive value")

    labels = None
    class_names = None
    samples = None
    logits_sum = None
    run_metrics = []
    for run, weight in zip(args.runs, weights):
        payload = torch.load(run / "test_logits.pt", map_location="cpu", weights_only=False)
        logits = payload[args.logits_key].float()
        if labels is None:
            labels = payload["labels"].long()
            class_names = payload["class_names"]
            samples = payload["samples"]
            logits_sum = torch.zeros_like(logits)
        elif not torch.equal(labels, payload["labels"].long()):
            raise ValueError(f"Label order mismatch in {run}")
        logits_sum += weight * logits
        run_accuracy = float((logits.argmax(dim=1) == labels).float().mean().item())
        run_metrics.append(
            {
                "run": str(run),
                "weight": weight,
                "accuracy": run_accuracy,
            }
        )

    assert labels is not None
    assert class_names is not None
    assert samples is not None
    assert logits_sum is not None

    logits_mean = logits_sum / sum(weights)
    predictions = logits_mean.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    matrix = confusion_matrix(labels, predictions, len(class_names))
    metrics = {
        "accuracy": accuracy,
        "logits_key": args.logits_key,
        "runs": run_metrics,
        "weights": weights,
        "class_names": class_names,
        "per_class_accuracy": per_class_accuracy(matrix, class_names),
        "confusion_matrix": matrix,
        "num_samples": int(labels.shape[0]),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(f"accuracy={accuracy:.4f}")
    print(f"saved_metrics={args.output}")


if __name__ == "__main__":
    main()
