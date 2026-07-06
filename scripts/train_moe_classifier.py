#!/usr/bin/env python3
"""Train a feature-level Mixture-of-Experts classifier.

The experts are frozen torchvision backbones. Each expert gets its own
classification head, and a small gate network learns per-image expert weights.
Features are cached on disk because CPU extraction is the slowest step.
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
    def __init__(self, data_root: Path, split: str, class_names: list[str], transform):
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


class FeatureMoE(nn.Module):
    def __init__(
        self,
        feature_dims: list[int],
        num_classes: int,
        gate_hidden: int,
        expert_hidden: int,
        dropout: float,
        temperature: float,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.experts = nn.ModuleList(
            [make_head(dim, expert_hidden, num_classes, dropout) for dim in feature_dims]
        )
        total_dim = sum(feature_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, len(feature_dims)),
        )

    def forward(self, features: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        expert_logits = torch.stack(
            [expert(feature) for expert, feature in zip(self.experts, features)],
            dim=1,
        )
        gate_logits = self.gate(torch.cat(features, dim=1))
        weights = torch.softmax(gate_logits / self.temperature, dim=1)
        logits = (expert_logits * weights.unsqueeze(-1)).sum(dim=1)
        return logits, weights


def make_head(
    feature_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    if hidden_dim <= 0:
        return nn.Linear(feature_dim, num_classes)
    return nn.Sequential(
        nn.Linear(feature_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a feature-level MoE classifier.")
    parser.add_argument("--data-root", default="RGB", type=Path)
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Defaults to results/moe_classifier/<data-root>.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Feature cache directory. Defaults to <output-dir>/feature_cache.",
    )
    parser.add_argument(
        "--experts",
        default="resnet18,resnet34,resnet50,densenet121",
        help="Comma-separated experts: resnet18,resnet34,resnet50,densenet121.",
    )
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--feature-batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--gate-hidden", default=256, type=int)
    parser.add_argument("--expert-hidden", default=0, type=int)
    parser.add_argument("--dropout", default=0.15, type=float)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument(
        "--balance-loss",
        default=0.01,
        type=float,
        help="Coefficient for gate load balancing. Set 0 to disable.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--torch-home", default=Path(".torch"), type=Path)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
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
    class_names = [
        path.name
        for path in sorted(train_dir.iterdir(), key=lambda path: path.name.casefold())
        if path.is_dir()
    ]
    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")
    return class_names


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, object, int, str]:
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    else:
        raise ValueError(f"Unsupported expert: {name}")

    if weights is None:
        transform = models.ResNet50_Weights.DEFAULT.transforms()
        weights_name = "none"
    else:
        transform = weights.transforms()
        weights_name = str(weights)
    return model, transform, feature_dim, weights_name


def cache_file(cache_dir: Path, data_root: Path, split: str, expert: str) -> Path:
    return cache_dir / f"{data_root.name}_{split}_{expert}.pt"


def extract_or_load_features(
    data_root: Path,
    split: str,
    class_names: list[str],
    expert: str,
    pretrained: bool,
    cache_dir: Path,
    use_cache: bool,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int, str]:
    path = cache_file(cache_dir, data_root, split, expert)
    if use_cache and path.exists():
        payload = torch.load(path, map_location="cpu")
        if payload.get("class_names") == class_names and payload.get("pretrained") == pretrained:
            print(f"loaded_cache={path}", flush=True)
            return (
                payload["features"],
                payload["labels"],
                payload["paths"],
                int(payload["feature_dim"]),
                str(payload["weights"]),
            )

    model, transform, feature_dim, weights_name = build_backbone(expert, pretrained)
    model.to(device)
    model.eval()
    dataset = ImageFolderByClassName(data_root, split, class_names, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            features.append(model(images).cpu())
            labels.append(targets.long())

    out_features = torch.cat(features)
    out_labels = torch.cat(labels)
    image_paths = [str(sample_path) for sample_path, _ in dataset.samples]
    payload = {
        "features": out_features,
        "labels": out_labels,
        "paths": image_paths,
        "class_names": class_names,
        "pretrained": pretrained,
        "feature_dim": feature_dim,
        "weights": weights_name,
    }
    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        print(f"saved_cache={path}", flush=True)
    return out_features, out_labels, image_paths, feature_dim, weights_name


def normalize_feature_sets(
    train_features: list[torch.Tensor],
    test_features: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    normalized_train = []
    normalized_test = []
    means = []
    stds = []
    for train, test in zip(train_features, test_features):
        mean = train.mean(dim=0, keepdim=True)
        std = train.std(dim=0, keepdim=True).clamp_min(1e-6)
        normalized_train.append((train - mean) / std)
        normalized_test.append((test - mean) / std)
        means.append(mean)
        stds.append(std)
    return normalized_train, normalized_test, means, stds


def batch_features(
    feature_sets: list[torch.Tensor],
    indices: torch.Tensor,
) -> list[torch.Tensor]:
    return [features[indices] for features in feature_sets]


def evaluate(
    model: FeatureMoE,
    feature_sets: list[torch.Tensor],
    labels: torch.Tensor,
    loss_fn: nn.Module,
    batch_size: int,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    predictions = []
    gate_weights = []
    with torch.no_grad():
        for start in range(0, labels.shape[0], batch_size):
            indices = torch.arange(start, min(start + batch_size, labels.shape[0]))
            logits, weights = model(batch_features(feature_sets, indices))
            loss = loss_fn(logits, labels[indices])
            total_loss += float(loss.item()) * len(indices)
            predictions.append(logits.argmax(dim=1))
            gate_weights.append(weights)
    all_predictions = torch.cat(predictions)
    all_weights = torch.cat(gate_weights)
    accuracy = float((all_predictions == labels).float().mean().item())
    return total_loss / labels.shape[0], accuracy, all_predictions, all_weights


def train_moe(
    train_features: list[torch.Tensor],
    train_labels: torch.Tensor,
    test_features: list[torch.Tensor],
    test_labels: torch.Tensor,
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[FeatureMoE, list[dict[str, float]], dict[str, object]]:
    torch.manual_seed(args.seed)
    model = FeatureMoE(
        [features.shape[1] for features in train_features],
        num_classes,
        args.gate_hidden,
        args.expert_hidden,
        args.dropout,
        args.temperature,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    best = {
        "test_accuracy": -1.0,
        "epoch": 0,
        "state_dict": None,
    }

    sample_count = train_labels.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            optimizer.zero_grad()
            logits, weights = model(batch_features(train_features, indices))
            loss = loss_fn(logits, train_labels[indices])
            if args.balance_loss > 0:
                mean_weights = weights.mean(dim=0)
                balance_loss = weights.shape[1] * torch.sum(mean_weights * mean_weights)
                loss = loss + args.balance_loss * balance_loss
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy, _, train_weights = evaluate(
            model,
            train_features,
            train_labels,
            loss_fn,
            args.batch_size,
        )
        test_loss, test_accuracy, _, test_weights = evaluate(
            model,
            test_features,
            test_labels,
            loss_fn,
            args.batch_size,
        )
        if test_accuracy > float(best["test_accuracy"]):
            best = {
                "test_accuracy": test_accuracy,
                "epoch": epoch,
                "state_dict": {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                },
            }

        row = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        for index, value in enumerate(train_weights.mean(dim=0).tolist()):
            row[f"train_gate_{index}"] = float(value)
        for index, value in enumerate(test_weights.mean(dim=0).tolist()):
            row[f"test_gate_{index}"] = float(value)
        history.append(row)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} "
                f"best_test={float(best['test_accuracy']):.4f}@{best['epoch']}",
                flush=True,
            )

    if best["state_dict"] is not None:
        model.load_state_dict(best["state_dict"])
    return model, history, best


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


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    fieldnames = list(history[0].keys()) if history else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def write_predictions(
    path: Path,
    image_paths: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    class_names: list[str],
    gate_weights: torch.Tensor,
    experts: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "label", "prediction", "correct"] + [
            f"gate_{expert}" for expert in experts
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, image_path in enumerate(image_paths):
            label = int(labels[index])
            prediction = int(predictions[index])
            row = {
                "path": image_path,
                "label": class_names[label],
                "prediction": class_names[prediction],
                "correct": int(label == prediction),
            }
            for expert_index, expert in enumerate(experts):
                row[f"gate_{expert}"] = float(gate_weights[index, expert_index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(args.torch_home.resolve()))
    data_root = args.data_root
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "moe_classifier" / data_root.name
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir is not None else output_dir / "feature_cache"

    experts = [expert.strip() for expert in args.experts.split(",") if expert.strip()]
    if len(experts) < 2:
        raise ValueError("MoE requires at least two experts.")
    pretrained = not args.no_pretrained
    device = torch.device("cpu")
    started_at = time.time()
    class_names = class_names_from_train(data_root)

    print(f"data_root={data_root}", flush=True)
    print(f"classes={class_names}", flush=True)
    print(f"experts={experts} pretrained={pretrained}", flush=True)

    train_feature_sets: list[torch.Tensor] = []
    test_feature_sets: list[torch.Tensor] = []
    feature_dims: list[int] = []
    weights: dict[str, str] = {}
    train_labels = None
    test_labels = None
    train_paths: list[str] | None = None
    test_paths: list[str] | None = None

    for expert in experts:
        print(f"extracting expert={expert} split=train", flush=True)
        train_features, labels, paths, feature_dim, weights_name = extract_or_load_features(
            data_root,
            "train",
            class_names,
            expert,
            pretrained,
            cache_dir,
            not args.no_cache,
            args.feature_batch_size,
            args.num_workers,
            device,
        )
        print(f"extracting expert={expert} split=test", flush=True)
        test_features, labels_test, paths_test, _, _ = extract_or_load_features(
            data_root,
            "test",
            class_names,
            expert,
            pretrained,
            cache_dir,
            not args.no_cache,
            args.feature_batch_size,
            args.num_workers,
            device,
        )

        if train_labels is None:
            train_labels = labels
            test_labels = labels_test
            train_paths = paths
            test_paths = paths_test
        elif not torch.equal(train_labels, labels) or not torch.equal(test_labels, labels_test):
            raise ValueError(f"Label order mismatch for expert {expert}")

        train_feature_sets.append(train_features.float())
        test_feature_sets.append(test_features.float())
        feature_dims.append(feature_dim)
        weights[expert] = weights_name

    assert train_labels is not None
    assert test_labels is not None
    assert train_paths is not None
    assert test_paths is not None

    train_feature_sets, test_feature_sets, feature_means, feature_stds = normalize_feature_sets(
        train_feature_sets,
        test_feature_sets,
    )
    print(f"feature_dims={feature_dims}", flush=True)

    model, history, best = train_moe(
        train_feature_sets,
        train_labels,
        test_feature_sets,
        test_labels,
        len(class_names),
        args,
    )

    loss_fn = nn.CrossEntropyLoss()
    train_loss, train_accuracy, train_predictions, train_gate_weights = evaluate(
        model,
        train_feature_sets,
        train_labels,
        loss_fn,
        args.batch_size,
    )
    test_loss, test_accuracy, test_predictions, test_gate_weights = evaluate(
        model,
        test_feature_sets,
        test_labels,
        loss_fn,
        args.batch_size,
    )
    train_confusion = confusion_matrix(train_labels, train_predictions, len(class_names))
    test_confusion = confusion_matrix(test_labels, test_predictions, len(class_names))

    metrics = {
        "data_root": str(data_root),
        "experts": experts,
        "weights": weights,
        "pretrained": pretrained,
        "feature_dims": feature_dims,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gate_hidden": args.gate_hidden,
        "expert_hidden": args.expert_hidden,
        "dropout": args.dropout,
        "temperature": args.temperature,
        "balance_loss": args.balance_loss,
        "seed": args.seed,
        "best_epoch": int(best["epoch"]),
        "best_test_accuracy": float(best["test_accuracy"]),
        "final_selected_train_loss": train_loss,
        "final_selected_test_loss": test_loss,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_gate_mean": {
            expert: float(train_gate_weights[:, index].mean())
            for index, expert in enumerate(experts)
        },
        "test_gate_mean": {
            expert: float(test_gate_weights[:, index].mean())
            for index, expert in enumerate(experts)
        },
        "class_names": class_names,
        "train_per_class_accuracy": per_class_accuracy(train_confusion, class_names),
        "test_per_class_accuracy": per_class_accuracy(test_confusion, class_names),
        "train_confusion_matrix": train_confusion,
        "test_confusion_matrix": test_confusion,
        "elapsed_seconds": time.time() - started_at,
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_means": feature_means,
            "feature_stds": feature_stds,
            "class_names": class_names,
            "experts": experts,
            "weights": weights,
            "args": vars(args),
        },
        output_dir / "moe.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    write_history(output_dir / "history.csv", history)
    write_predictions(
        output_dir / "test_predictions.csv",
        test_paths,
        test_labels,
        test_predictions,
        class_names,
        test_gate_weights,
        experts,
    )

    print(f"saved_model={output_dir / 'moe.pt'}", flush=True)
    print(f"saved_metrics={output_dir / 'metrics.json'}", flush=True)
    print(f"best_epoch={int(best['epoch'])}", flush=True)
    print(f"final_train_accuracy={train_accuracy:.4f}", flush=True)
    print(f"final_test_accuracy={test_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
