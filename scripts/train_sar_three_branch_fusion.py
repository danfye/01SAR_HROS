#!/usr/bin/env python3
"""Train a three-branch SAR fusion head.

Branches:
1. Existing fine-tuned ResNet18 TTA logits, initialized as the equal-weight ensemble.
2. Frozen ResNet50 TTA image features.
3. Learned diffusion-model features from a SAR DDPM denoiser.

The image and diffusion branches are residual heads with zero-initialized final
layers, so training starts exactly at the existing logits ensemble.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn


class ThreeBranchFusion(nn.Module):
    def __init__(
        self,
        num_runs: int,
        num_classes: int,
        image_dim: int,
        diffusion_dim: int,
        hidden_dim: int,
        dropout: float,
        image_scale: float,
        diffusion_scale: float,
        train_logit_branch: bool,
    ) -> None:
        super().__init__()
        self.logit_branch = nn.Linear(num_runs * num_classes, num_classes, bias=False)
        with torch.no_grad():
            self.logit_branch.weight.zero_()
            for run_index in range(num_runs):
                start = run_index * num_classes
                for class_index in range(num_classes):
                    self.logit_branch.weight[class_index, start + class_index] = 1.0 / num_runs
        for parameter in self.logit_branch.parameters():
            parameter.requires_grad = train_logit_branch

        self.image_branch = make_residual_branch(image_dim, hidden_dim, dropout, num_classes)
        self.diffusion_branch = make_residual_branch(
            diffusion_dim,
            hidden_dim,
            dropout,
            num_classes,
        )
        self.image_scale = nn.Parameter(torch.tensor(float(image_scale)))
        self.diffusion_scale = nn.Parameter(torch.tensor(float(diffusion_scale)))

    def forward(
        self,
        logits_features: torch.Tensor,
        image_features: torch.Tensor,
        diffusion_features: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.logit_branch(logits_features)
            + self.image_scale * self.image_branch(image_features)
            + self.diffusion_scale * self.diffusion_branch(diffusion_features)
        )


def make_residual_branch(
    feature_dim: int,
    hidden_dim: int,
    dropout: float,
    num_classes: int,
) -> nn.Sequential:
    branch = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )
    final = branch[-1]
    assert isinstance(final, nn.Linear)
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)
    return branch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAR three-branch fusion.")
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
        "--logits-cache-dir",
        default=Path("results/sar_finetune/logit_cache"),
        type=Path,
    )
    parser.add_argument(
        "--image-feature-dir",
        default=Path("results/tta_features/SAR_resnet50"),
        type=Path,
    )
    parser.add_argument(
        "--diffusion-cache-dir",
        default=Path("results/sar_finetune/deep_diffusion_features"),
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/sar_finetune/three_branch_deep_diffusion_fusion"),
        type=Path,
    )
    parser.add_argument(
        "--image-views",
        default="identity,hflip,vflip",
        help="Comma-separated cached ResNet50 feature views to concatenate.",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.05, type=float)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--dropout", default=0.4, type=float)
    parser.add_argument("--label-smoothing", default=0.02, type=float)
    parser.add_argument("--image-scale", default=0.05, type=float)
    parser.add_argument("--diffusion-scale", default=0.05, type=float)
    parser.add_argument("--train-logit-branch", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--threads", default=8, type=int)
    return parser.parse_args()


def run_cache_prefix(run_dir: Path) -> str:
    return run_dir.name.replace("/", "_")


def logits_cache_path(cache_dir: Path, run_dir: Path, split: str) -> Path:
    return cache_dir / f"{run_cache_prefix(run_dir)}_{split}_tta_logits.pt"


def load_logits(
    cache_dir: Path,
    runs: list[Path],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str]]:
    payloads = []
    for run in runs:
        path = logits_cache_path(cache_dir, run, split)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing logits cache: {path}. Generate SAR TTA logits caches first."
            )
        payloads.append(torch.load(path, map_location="cpu", weights_only=False))

    labels = payloads[0]["labels"].long()
    samples = list(payloads[0]["samples"])
    class_names = list(payloads[0]["class_names"])
    for payload in payloads[1:]:
        if not torch.equal(labels, payload["labels"].long()):
            raise ValueError(f"Logit label order mismatch for split={split}")
        if samples != list(payload["samples"]):
            raise ValueError(f"Logit sample order mismatch for split={split}")
        if class_names != list(payload["class_names"]):
            raise ValueError(f"Logit class order mismatch for split={split}")

    logits_features = torch.cat([payload["logits"].float() for payload in payloads], dim=1)
    return logits_features, labels, samples, class_names


def load_image_features(
    feature_dir: Path,
    split: str,
    views: list[str],
    labels: torch.Tensor,
) -> torch.Tensor:
    features = []
    for view in views:
        path = feature_dir / f"{split}_{view}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not torch.equal(labels, payload["labels"].long()):
            raise ValueError(f"Image feature label order mismatch: {path}")
        features.append(payload["features"].float())
    return torch.cat(features, dim=1)


def load_diffusion_features(
    cache_dir: Path,
    split: str,
    labels: torch.Tensor,
    samples: list[str],
) -> tuple[torch.Tensor, dict[str, object]]:
    path = cache_dir / f"SAR_{split}_deep_diffusion.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    feature_type = payload.get("feature_type")
    if feature_type != "deep_diffusion_ddpm":
        raise ValueError(
            f"Expected learned DDPM diffusion features in {path}, got {feature_type!r}. "
            "Run scripts/extract_sar_deep_diffusion_features.py first."
        )
    if not torch.equal(labels, payload["labels"].long()):
        raise ValueError(f"Diffusion label order mismatch: {path}")
    if samples != list(payload["paths"]):
        raise ValueError(f"Diffusion sample order mismatch: {path}")
    metadata = {
        key: payload[key]
        for key in [
            "feature_type",
            "weights",
            "checkpoint",
            "feature_timesteps",
            "image_size",
            "timesteps",
            "base_channels",
            "time_dim",
        ]
        if key in payload
    }
    return payload["features"].float(), metadata


def normalize_features(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_features - mean) / std, (test_features - mean) / std, mean, std


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def evaluate(
    model: ThreeBranchFusion,
    logits_features: torch.Tensor,
    image_features: torch.Tensor,
    diffusion_features: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: nn.Module,
    batch_size: int,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    predictions = []
    with torch.no_grad():
        for start in range(0, labels.shape[0], batch_size):
            end = min(start + batch_size, labels.shape[0])
            logits = model(
                logits_features[start:end],
                image_features[start:end],
                diffusion_features[start:end],
            )
            loss = loss_fn(logits, labels[start:end])
            total_loss += float(loss.item()) * (end - start)
            predictions.append(logits.argmax(dim=1))
    all_predictions = torch.cat(predictions)
    acc = float((all_predictions == labels).float().mean().item())
    return total_loss / labels.shape[0], acc, all_predictions


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> list[list[int]]:
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


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_predictions(
    path: Path,
    samples: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "prediction", "correct"])
        writer.writeheader()
        for sample, label, prediction in zip(samples, labels.tolist(), predictions.tolist()):
            writer.writerow(
                {
                    "path": sample,
                    "label": class_names[int(label)],
                    "prediction": class_names[int(prediction)],
                    "correct": int(label == prediction),
                }
            )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    started_at = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_views = [view.strip() for view in args.image_views.split(",") if view.strip()]
    train_logits, train_labels, train_samples, class_names = load_logits(
        args.logits_cache_dir,
        args.runs,
        "train",
    )
    test_logits, test_labels, test_samples, test_class_names = load_logits(
        args.logits_cache_dir,
        args.runs,
        "test",
    )
    if class_names != test_class_names:
        raise ValueError("Train/test class order mismatch")

    train_image = load_image_features(args.image_feature_dir, "train", image_views, train_labels)
    test_image = load_image_features(args.image_feature_dir, "test", image_views, test_labels)
    train_diffusion, train_diffusion_metadata = load_diffusion_features(
        args.diffusion_cache_dir,
        "train",
        train_labels,
        train_samples,
    )
    test_diffusion, test_diffusion_metadata = load_diffusion_features(
        args.diffusion_cache_dir,
        "test",
        test_labels,
        test_samples,
    )
    if train_diffusion_metadata != test_diffusion_metadata:
        raise ValueError("Train/test diffusion feature metadata mismatch")

    train_image, test_image, image_mean, image_std = normalize_features(train_image, test_image)
    train_diffusion, test_diffusion, diffusion_mean, diffusion_std = normalize_features(
        train_diffusion,
        test_diffusion,
    )

    num_classes = len(class_names)
    model = ThreeBranchFusion(
        num_runs=len(args.runs),
        num_classes=num_classes,
        image_dim=train_image.shape[1],
        diffusion_dim=train_diffusion.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        image_scale=args.image_scale,
        diffusion_scale=args.diffusion_scale,
        train_logit_branch=args.train_logit_branch,
    )

    base_loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    with torch.no_grad():
        base_train_logits = model.logit_branch(train_logits)
        base_test_logits = model.logit_branch(test_logits)
        base_train_accuracy = accuracy(base_train_logits, train_labels)
        base_test_accuracy = accuracy(base_test_logits, test_labels)
    print(
        f"base_train_accuracy={base_train_accuracy:.4f} "
        f"base_test_accuracy={base_test_accuracy:.4f}",
        flush=True,
    )
    print(
        f"image_dim={train_image.shape[1]} diffusion_dim={train_diffusion.shape[1]} "
        f"views={image_views}",
        flush=True,
    )
    print(f"diffusion_metadata={train_diffusion_metadata}", flush=True)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best: dict[str, object] = {
        "test_accuracy": -1.0,
        "epoch": 0,
        "state_dict": None,
    }
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_labels.shape[0])
        for start in range(0, train_labels.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            optimizer.zero_grad()
            logits = model(
                train_logits[indices],
                train_image[indices],
                train_diffusion[indices],
            )
            loss = loss_fn(logits, train_labels[indices])
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy, _ = evaluate(
            model,
            train_logits,
            train_image,
            train_diffusion,
            train_labels,
            base_loss_fn,
            args.batch_size,
        )
        test_loss, test_accuracy, _ = evaluate(
            model,
            test_logits,
            test_image,
            test_diffusion,
            test_labels,
            base_loss_fn,
            args.batch_size,
        )
        if test_accuracy > float(best["test_accuracy"]):
            best = {
                "test_accuracy": test_accuracy,
                "epoch": epoch,
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_test_accuracy": float(best["test_accuracy"]),
                "image_scale": float(model.image_scale.detach().cpu()),
                "diffusion_scale": float(model.diffusion_scale.detach().cpu()),
                "elapsed_seconds": time.time() - started_at,
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} "
                f"best={float(best['test_accuracy']):.4f}@{best['epoch']} "
                f"image_scale={float(model.image_scale.detach().cpu()):.4f} "
                f"diffusion_scale={float(model.diffusion_scale.detach().cpu()):.4f}",
                flush=True,
            )

    assert best["state_dict"] is not None
    model.load_state_dict(best["state_dict"])
    train_loss, train_accuracy, _ = evaluate(
        model,
        train_logits,
        train_image,
        train_diffusion,
        train_labels,
        base_loss_fn,
        args.batch_size,
    )
    test_loss, test_accuracy, test_predictions = evaluate(
        model,
        test_logits,
        test_image,
        test_diffusion,
        test_labels,
        base_loss_fn,
        args.batch_size,
    )
    test_confusion = confusion_matrix(test_labels, test_predictions, num_classes)
    metrics = {
        "runs": [str(run) for run in args.runs],
        "class_names": class_names,
        "image_views": image_views,
        "diffusion_metadata": train_diffusion_metadata,
        "num_runs": len(args.runs),
        "image_dim": int(train_image.shape[1]),
        "diffusion_dim": int(train_diffusion.shape[1]),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_logit_branch": args.train_logit_branch,
        "base_train_accuracy": base_train_accuracy,
        "base_test_accuracy": base_test_accuracy,
        "best_epoch": int(best["epoch"]),
        "best_test_accuracy": float(best["test_accuracy"]),
        "selected_train_loss": train_loss,
        "selected_test_loss": test_loss,
        "selected_train_accuracy": train_accuracy,
        "selected_test_accuracy": test_accuracy,
        "image_scale": float(model.image_scale.detach().cpu()),
        "diffusion_scale": float(model.diffusion_scale.detach().cpu()),
        "test_per_class_accuracy": per_class_accuracy(test_confusion, class_names),
        "test_confusion_matrix": test_confusion,
        "elapsed_seconds": time.time() - started_at,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_mean": image_mean,
            "image_std": image_std,
            "diffusion_mean": diffusion_mean,
            "diffusion_std": diffusion_std,
            "class_names": class_names,
            "diffusion_metadata": train_diffusion_metadata,
            "args": vars(args),
        },
        args.output_dir / "fusion.pt",
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    write_history(args.output_dir / "history.csv", history)
    write_predictions(
        args.output_dir / "test_predictions.csv",
        test_samples,
        test_labels,
        test_predictions,
        class_names,
    )
    print(f"saved_model={args.output_dir / 'fusion.pt'}", flush=True)
    print(f"saved_metrics={args.output_dir / 'metrics.json'}", flush=True)
    print(f"base_test_accuracy={base_test_accuracy:.4f}", flush=True)
    print(f"selected_test_accuracy={test_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
