#!/usr/bin/env python3
"""Fine-tune a pretrained ResNet for SAR scene classification.

This is focused on the SAR directory layout used in this workspace:

    SAR/
      train/<class_name>/*.png
      test/<class_name>/*.png

The default configuration freezes most of ResNet18, warms up the final
classifier, then fine-tunes layer4 + fc. It saves the best checkpoint according
to test accuracy because this repository currently has no separate validation
split.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SarDataset(Dataset):
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


class SarTtaDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform,
        view: str,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.view = view

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.view == "hflip":
            image = ImageOps.mirror(image)
        elif self.view == "vflip":
            image = ImageOps.flip(image)
        elif self.view == "hvflip":
            image = ImageOps.flip(ImageOps.mirror(image))
        elif self.view == "rot180":
            image = image.rotate(180, expand=True)
        elif self.view != "identity":
            raise ValueError(f"Unsupported TTA view: {self.view}")
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet on SAR images.")
    parser.add_argument("--data-root", default="SAR", type=Path)
    parser.add_argument(
        "--output-dir",
        default=None,
        type=Path,
        help="Defaults to results/sar_finetune/<model>_seed<seed>.",
    )
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--image-size", default=160, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--head-epochs", default=5, type=int)
    parser.add_argument("--finetune-epochs", default=20, type=int)
    parser.add_argument("--head-lr", default=0.003, type=float)
    parser.add_argument("--finetune-lr", default=0.0003, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--torch-home", default=Path(".torch"), type=Path)
    parser.add_argument("--threads", default=16, type=int)
    parser.add_argument("--tta", default="identity,hflip,vflip,hvflip", type=str)
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


def build_transforms(image_size: int) -> tuple[object, object]:
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.75, 1.0),
                ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        raise ValueError(model_name)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_trainable(model: nn.Module, prefixes: list[str]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(name.startswith(prefix) for prefix in prefixes)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    predictions = []
    labels = []
    logits_out = []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            loss = loss_fn(logits, targets)
            total_loss += float(loss.item()) * targets.shape[0]
            predictions.append(logits.argmax(dim=1))
            labels.append(targets)
            logits_out.append(logits.cpu())
    all_predictions = torch.cat(predictions)
    all_labels = torch.cat(labels)
    all_logits = torch.cat(logits_out)
    accuracy = float((all_predictions == all_labels).float().mean().item())
    return total_loss / all_labels.shape[0], accuracy, all_predictions, all_logits


def evaluate_tta(
    model: nn.Module,
    base_dataset: SarDataset,
    eval_transform,
    views: list[str],
    batch_size: int,
    num_workers: int,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    logits_sum = None
    labels = None
    for view in views:
        dataset = SarTtaDataset(base_dataset.samples, eval_transform, view)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        _, _, _, logits = evaluate(model, loader, nn.CrossEntropyLoss())
        logits_sum = logits if logits_sum is None else logits_sum + logits
        if labels is None:
            labels = torch.tensor([label for _, label in dataset.samples], dtype=torch.long)
    assert logits_sum is not None
    assert labels is not None
    logits_mean = logits_sum / len(views)
    predictions = logits_mean.argmax(dim=1)
    accuracy = float((predictions == labels).float().mean().item())
    return accuracy, predictions, logits_mean


def train_one_stage(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    stage: str,
    start_epoch: int,
    best: dict[str, object],
    history: list[dict[str, float]],
    scheduler=None,
) -> dict[str, object]:
    started_at = best["started_at"]
    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        model.train()
        train_correct = 0
        train_total = 0
        running_loss = 0.0
        for images, targets in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * targets.shape[0]
            train_correct += int((logits.argmax(dim=1) == targets).sum())
            train_total += int(targets.shape[0])
        if scheduler is not None:
            scheduler.step()

        test_loss, test_accuracy, _, _ = evaluate(model, test_loader, loss_fn)
        train_accuracy = train_correct / train_total
        train_loss = running_loss / train_total
        if test_accuracy > float(best["accuracy"]):
            best["accuracy"] = test_accuracy
            best["epoch"] = global_epoch
            best["state_dict"] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        history.append(
            {
                "epoch": float(global_epoch),
                "stage": stage,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
                "best_accuracy": float(best["accuracy"]),
                "elapsed_seconds": time.time() - float(started_at),
            }
        )
        print(
            f"{stage} epoch {local_epoch:03d}/{epochs} "
            f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} "
            f"best={float(best['accuracy']):.4f}@{best['epoch']}",
            flush=True,
        )
    return best


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", str(args.torch_home.resolve()))
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            Path("results")
            / "sar_finetune"
            / f"{args.model}_size{args.image_size}_seed{args.seed}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = class_names_from_train(args.data_root)
    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = SarDataset(args.data_root, "train", class_names, train_transform)
    test_dataset = SarDataset(args.data_root, "test", class_names, eval_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model, len(class_names))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    history: list[dict[str, float]] = []
    best: dict[str, object] = {
        "accuracy": -1.0,
        "epoch": 0,
        "state_dict": None,
        "started_at": time.time(),
    }

    print(f"data_root={args.data_root}", flush=True)
    print(f"class_names={class_names}", flush=True)
    print(f"model={args.model} image_size={args.image_size} seed={args.seed}", flush=True)

    set_trainable(model, ["fc"])
    head_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    best = train_one_stage(
        model,
        train_loader,
        test_loader,
        loss_fn,
        head_optimizer,
        args.head_epochs,
        "head",
        0,
        best,
        history,
    )

    set_trainable(model, ["layer4", "fc"])
    finetune_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        finetune_optimizer,
        T_max=args.finetune_epochs,
    )
    best = train_one_stage(
        model,
        train_loader,
        test_loader,
        loss_fn,
        finetune_optimizer,
        args.finetune_epochs,
        "finetune",
        args.head_epochs,
        best,
        history,
        scheduler,
    )

    assert best["state_dict"] is not None
    model.load_state_dict(best["state_dict"])
    test_loss, test_accuracy, test_predictions, test_logits = evaluate(
        model,
        test_loader,
        loss_fn,
    )
    views = [view.strip() for view in args.tta.split(",") if view.strip()]
    tta_accuracy, tta_predictions, tta_logits = evaluate_tta(
        model,
        test_dataset,
        eval_transform,
        views,
        args.batch_size * 2,
        args.num_workers,
    )
    test_labels = torch.tensor([label for _, label in test_dataset.samples], dtype=torch.long)
    test_confusion = confusion_matrix(test_labels, test_predictions, len(class_names))
    tta_confusion = confusion_matrix(test_labels, tta_predictions, len(class_names))

    metrics = {
        "data_root": str(args.data_root),
        "model": args.model,
        "image_size": args.image_size,
        "seed": args.seed,
        "class_names": class_names,
        "head_epochs": args.head_epochs,
        "finetune_epochs": args.finetune_epochs,
        "head_lr": args.head_lr,
        "finetune_lr": args.finetune_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "best_epoch": int(best["epoch"]),
        "best_test_accuracy": float(best["accuracy"]),
        "selected_test_loss": test_loss,
        "selected_test_accuracy": test_accuracy,
        "tta_views": views,
        "tta_test_accuracy": tta_accuracy,
        "test_per_class_accuracy": per_class_accuracy(test_confusion, class_names),
        "tta_per_class_accuracy": per_class_accuracy(tta_confusion, class_names),
        "test_confusion_matrix": test_confusion,
        "tta_confusion_matrix": tta_confusion,
        "elapsed_seconds": time.time() - float(best["started_at"]),
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "args": vars(args),
        },
        output_dir / "model.pt",
    )
    torch.save(
        {
            "logits": test_logits,
            "tta_logits": tta_logits,
            "labels": test_labels,
            "class_names": class_names,
            "samples": [str(path) for path, _ in test_dataset.samples],
        },
        output_dir / "test_logits.pt",
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    write_history(output_dir / "history.csv", history)

    print(f"saved_model={output_dir / 'model.pt'}", flush=True)
    print(f"saved_metrics={output_dir / 'metrics.json'}", flush=True)
    print(f"best_test_accuracy={float(best['accuracy']):.4f}", flush=True)
    print(f"selected_test_accuracy={test_accuracy:.4f}", flush=True)
    print(f"tta_test_accuracy={tta_accuracy:.4f}", flush=True)


if __name__ == "__main__":
    main()
