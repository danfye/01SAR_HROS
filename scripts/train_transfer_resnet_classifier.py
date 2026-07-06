#!/usr/bin/env python3
"""Train a transfer-learning image classifier with torchvision ResNet features.

This script uses an ImageNet-pretrained ResNet as a frozen feature extractor and
trains a lightweight linear classifier on top. It is intended for small image
classification datasets laid out as:

    RGB/
      train/<class_name>/*.png
      test/<class_name>/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageFolderByClassName(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        class_names: list[str],
        transform,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        split_dir = data_root / split

        for class_index, class_name in enumerate(class_names):
            class_dir = resolve_class_dir(split_dir, class_name)
            paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise ValueError(f"No images found in {class_dir}")
            self.samples.extend((path, class_index) for path in paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a ResNet transfer-learning classifier."
    )
    parser.add_argument(
        "--data-root",
        default="RGB",
        type=Path,
        help="Dataset root containing train/ and test/ directories.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Output directory. Defaults to results/transfer_resnet/<data-root>.",
    )
    parser.add_argument(
        "--model",
        default="resnet50",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Frozen ImageNet-pretrained backbone.",
    )
    parser.add_argument("--epochs", default=250, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--torch-home",
        default=Path(".torch"),
        type=Path,
        help="Directory for torchvision pretrained weight cache.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet weights. Accuracy will usually be much lower.",
    )
    return parser.parse_args()


def resolve_class_dir(split_dir: Path, class_name: str) -> Path:
    exact = split_dir / class_name
    if exact.exists():
        return exact

    for path in split_dir.iterdir():
        if path.is_dir() and path.name.casefold() == class_name.casefold():
            return path
    raise FileNotFoundError(f"Missing class {class_name!r} in {split_dir}")


def class_names_from_train(data_root: Path) -> list[str]:
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    class_names = [
        path.name
        for path in sorted(train_dir.iterdir(), key=lambda path: path.name.casefold())
        if path.is_dir()
    ]
    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")
    return class_names


def build_feature_extractor(
    model_name: str,
    pretrained: bool,
) -> tuple[nn.Module, object, int, str]:
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if weights is None:
        transform = models.ResNet50_Weights.DEFAULT.transforms()
        weights_name = "none"
    else:
        transform = weights.transforms()
        weights_name = str(weights)

    return model, transform, feature_dim, weights_name


def extract_features(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            features.append(model(images).cpu())
            labels.append(targets.long())

    return torch.cat(features), torch.cat(labels)


def normalize_features(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_features - mean) / std, (test_features - mean) / std, mean, std


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_classifier(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[nn.Linear, list[dict[str, float]]]:
    torch.manual_seed(args.seed)
    classifier = nn.Linear(train_features.shape[1], num_classes)
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        optimizer.zero_grad()
        train_logits = classifier(train_features)
        loss = loss_fn(train_logits, train_labels)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            train_logits = classifier(train_features)
            test_logits = classifier(test_features)
            train_loss = float(loss_fn(train_logits, train_labels).item())
            test_loss = float(loss_fn(test_logits, test_labels).item())
            train_accuracy = accuracy(train_logits, train_labels)
            test_accuracy = accuracy(test_logits, test_labels)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}",
                flush=True,
            )

    return classifier, history


def confusion_matrix(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(label)][int(prediction)] += 1
    return matrix


def per_class_accuracy(
    matrix: list[list[int]],
    class_names: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        total = sum(matrix[index])
        result[class_name] = matrix[index][index] / total if total else 0.0
    return result


def counts_by_class(dataset: ImageFolderByClassName, class_names: list[str]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_names}
    for _, label in dataset.samples:
        counts[class_names[label]] += 1
    return counts


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "test_loss",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def write_predictions(
    path: Path,
    dataset: ImageFolderByClassName,
    class_names: list[str],
    predictions: torch.Tensor,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "label", "prediction", "correct"],
        )
        writer.writeheader()
        for (image_path, label), prediction in zip(dataset.samples, predictions.tolist()):
            writer.writerow(
                {
                    "path": str(image_path),
                    "label": class_names[label],
                    "prediction": class_names[int(prediction)],
                    "correct": int(label == int(prediction)),
                }
            )


def main() -> None:
    args = parse_args()
    torch_home = args.torch_home.resolve()
    os.environ.setdefault("TORCH_HOME", str(torch_home))

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "transfer_resnet" / args.data_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    device = torch.device("cpu")
    pretrained = not args.no_pretrained
    class_names = class_names_from_train(args.data_root)

    print(f"data_root={args.data_root}")
    print(f"classes={class_names}")
    print(f"model={args.model} pretrained={pretrained}")
    print("building feature extractor...")
    feature_extractor, transform, feature_dim, weights_name = build_feature_extractor(
        args.model,
        pretrained,
    )
    feature_extractor.to(device)

    train_dataset = ImageFolderByClassName(
        args.data_root,
        "train",
        class_names,
        transform,
    )
    test_dataset = ImageFolderByClassName(
        args.data_root,
        "test",
        class_names,
        transform,
    )

    print("extracting train features...")
    train_features, train_labels = extract_features(
        feature_extractor,
        train_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )
    print("extracting test features...")
    test_features, test_labels = extract_features(
        feature_extractor,
        test_dataset,
        args.batch_size,
        args.num_workers,
        device,
    )
    print(f"train_features={tuple(train_features.shape)}")
    print(f"test_features={tuple(test_features.shape)}")

    train_features, test_features, feature_mean, feature_std = normalize_features(
        train_features,
        test_features,
    )
    classifier, history = train_classifier(
        train_features,
        train_labels,
        test_features,
        test_labels,
        len(class_names),
        args,
    )

    classifier.eval()
    with torch.no_grad():
        train_logits = classifier(train_features)
        test_logits = classifier(test_features)
        train_predictions = train_logits.argmax(dim=1)
        test_predictions = test_logits.argmax(dim=1)
        train_accuracy = accuracy(train_logits, train_labels)
        test_accuracy = accuracy(test_logits, test_labels)

    train_confusion = confusion_matrix(
        train_labels,
        train_predictions,
        len(class_names),
    )
    test_confusion = confusion_matrix(
        test_labels,
        test_predictions,
        len(class_names),
    )

    metrics = {
        "data_root": str(args.data_root),
        "model": args.model,
        "weights": weights_name,
        "pretrained": pretrained,
        "feature_dim": feature_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "class_names": class_names,
        "train_counts": counts_by_class(train_dataset, class_names),
        "test_counts": counts_by_class(test_dataset, class_names),
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_per_class_accuracy": per_class_accuracy(train_confusion, class_names),
        "test_per_class_accuracy": per_class_accuracy(test_confusion, class_names),
        "train_confusion_matrix": train_confusion,
        "test_confusion_matrix": test_confusion,
        "elapsed_seconds": time.time() - started_at,
    }

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "class_names": class_names,
            "model": args.model,
            "weights": weights_name,
            "pretrained": pretrained,
        },
        output_dir / "classifier.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    write_history(output_dir / "history.csv", history)
    write_predictions(
        output_dir / "test_predictions.csv",
        test_dataset,
        class_names,
        test_predictions,
    )

    print(f"saved_classifier={output_dir / 'classifier.pt'}")
    print(f"saved_metrics={output_dir / 'metrics.json'}")
    print(f"final_train_accuracy={train_accuracy:.4f}")
    print(f"final_test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
