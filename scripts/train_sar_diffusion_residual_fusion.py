#!/usr/bin/env python3
"""Train a residual fusion head using SAR fine-tuned logits plus diffusion features.

The logit branch starts as the equal-weight ensemble of saved SAR ResNet18 TTA
runs. A small diffusion-statistics branch is trained as a residual correction,
so the model starts from the strong existing ensemble instead of replacing it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SarTtaDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        class_names: list[str],
        transform,
        view: str,
    ) -> None:
        self.transform = transform
        self.view = view
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


class ResidualFusion(nn.Module):
    def __init__(
        self,
        num_runs: int,
        num_classes: int,
        diffusion_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_scale: float,
        train_logit_branch: bool,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.logit_branch = nn.Linear(num_runs * num_classes, num_classes, bias=False)
        with torch.no_grad():
            self.logit_branch.weight.zero_()
            for run_index in range(num_runs):
                start = run_index * num_classes
                for class_index in range(num_classes):
                    self.logit_branch.weight[class_index, start + class_index] = 1.0 / num_runs
        for parameter in self.logit_branch.parameters():
            parameter.requires_grad = train_logit_branch

        self.diffusion_branch = nn.Sequential(
            nn.LayerNorm(diffusion_dim),
            nn.Linear(diffusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        final = self.diffusion_branch[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, run_logits: torch.Tensor, diffusion_features: torch.Tensor) -> torch.Tensor:
        base_logits = self.logit_branch(run_logits)
        residual = self.diffusion_branch(diffusion_features)
        return base_logits + self.residual_scale * residual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAR residual fusion with diffusion features."
    )
    parser.add_argument("--data-root", default=Path("SAR"), type=Path)
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
        "--output-dir",
        default=Path("results/sar_finetune/diffusion_residual_fusion"),
        type=Path,
    )
    parser.add_argument(
        "--logits-cache-dir",
        default=Path("results/sar_finetune/logit_cache"),
        type=Path,
    )
    parser.add_argument(
        "--diffusion-cache-dir",
        default=Path("results/moe_classifier/SAR/feature_cache"),
        type=Path,
    )
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-2, type=float)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--dropout", default=0.35, type=float)
    parser.add_argument("--label-smoothing", default=0.02, type=float)
    parser.add_argument("--residual-scale", default=0.1, type=float)
    parser.add_argument(
        "--train-logit-branch",
        action="store_true",
        help="Also train the logit ensemble weights. Default keeps them fixed.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
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


def eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
    else:
        raise ValueError(f"Unsupported saved model: {model_name}")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def views_from_run(run_dir: Path) -> list[str]:
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        views = metrics.get("tta_views")
        if views:
            return [str(view) for view in views]
    return ["identity", "hflip", "vflip", "hvflip"]


def run_cache_prefix(run_dir: Path) -> str:
    return run_dir.name.replace("/", "_")


def logits_cache_path(cache_dir: Path, run_dir: Path, split: str) -> Path:
    return cache_dir / f"{run_cache_prefix(run_dir)}_{split}_tta_logits.pt"


def load_existing_test_logits(run_dir: Path, class_names: list[str]) -> dict[str, object] | None:
    path = run_dir / "test_logits.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("class_names") != class_names:
        return None
    return {
        "logits": payload["tta_logits"].float(),
        "labels": payload["labels"].long(),
        "samples": payload["samples"],
        "class_names": payload["class_names"],
    }


def compute_or_load_tta_logits(
    run_dir: Path,
    data_root: Path,
    split: str,
    class_names: list[str],
    cache_dir: Path,
    batch_size: int,
    num_workers: int,
) -> dict[str, object]:
    cache_path = logits_cache_path(cache_dir, run_dir, split)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("class_names") == class_names:
            print(f"loaded_logits_cache={cache_path}", flush=True)
            return payload

    if split == "test":
        payload = load_existing_test_logits(run_dir, class_names)
        if payload is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_path)
            print(f"saved_logits_cache={cache_path}", flush=True)
            return payload

    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=False)
    run_args = checkpoint.get("args", {})
    image_size = int(run_args.get("image_size", 160))
    model_name = str(run_args.get("model", "resnet18"))
    model = build_model(model_name, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    views = views_from_run(run_dir)
    logits_sum = None
    labels = None
    samples = None
    transform = eval_transform(image_size)
    for view in views:
        dataset = SarTtaDataset(data_root, split, class_names, transform, view)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        view_logits = []
        with torch.no_grad():
            for images, _targets in loader:
                view_logits.append(model(images).cpu())
        logits = torch.cat(view_logits)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        if labels is None:
            labels = torch.tensor([label for _path, label in dataset.samples], dtype=torch.long)
            samples = [str(path) for path, _label in dataset.samples]

    assert logits_sum is not None
    assert labels is not None
    assert samples is not None
    payload = {
        "logits": logits_sum / len(views),
        "labels": labels,
        "samples": samples,
        "class_names": class_names,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"saved_logits_cache={cache_path}", flush=True)
    return payload


def heat_diffuse(array: np.ndarray, steps: int, rate: float = 0.18) -> np.ndarray:
    out = array.astype(np.float32, copy=True)
    for _ in range(steps):
        padded = np.pad(out, 1, mode="reflect")
        laplacian = (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            - 4.0 * out
        )
        out = np.clip(out + rate * laplacian, 0.0, 1.0)
    return out


def pooled_map_features(array: np.ndarray, pooled_size: int) -> np.ndarray:
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), mode="L")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    pooled = image.resize((pooled_size, pooled_size), resampling)
    return np.asarray(pooled, dtype=np.float32).reshape(-1) / 255.0


def map_summary_features(array: np.ndarray) -> np.ndarray:
    gx = np.diff(array, axis=1)
    gy = np.diff(array, axis=0)
    gradient_energy = float(np.mean(gx * gx) + np.mean(gy * gy))
    quantiles = np.quantile(array, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32)
    summary = np.asarray(
        [
            float(array.mean()),
            float(array.std()),
            float(array.min()),
            float(array.max()),
            gradient_energy,
        ],
        dtype=np.float32,
    )
    return np.concatenate([summary, quantiles])


def diffusion_feature_from_path(path: Path, image_size: int = 64, pooled_size: int = 16) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    with Image.open(path) as image:
        image = image.convert("L").resize((image_size, image_size), resampling)
        base = np.asarray(image, dtype=np.float32) / 255.0

    feature_blocks = []
    previous = base
    for steps in [0, 1, 2, 4, 8, 16]:
        diffused = base if steps == 0 else heat_diffuse(base, steps)
        feature_blocks.append(pooled_map_features(diffused, pooled_size))
        feature_blocks.append(map_summary_features(diffused))
        if steps > 0:
            residual = np.abs(previous - diffused)
            feature_blocks.append(pooled_map_features(residual, pooled_size))
            feature_blocks.append(map_summary_features(residual))
        previous = diffused
    return np.concatenate(feature_blocks).astype(np.float32)


def diffusion_cache_path(cache_dir: Path, data_root: Path, split: str) -> Path:
    return cache_dir / f"{data_root.name}_{split}_diffusion.pt"


def compute_or_load_diffusion_features(
    data_root: Path,
    split: str,
    class_names: list[str],
    cache_dir: Path,
) -> dict[str, object]:
    cache_path = diffusion_cache_path(cache_dir, data_root, split)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if payload.get("class_names") == class_names:
            print(f"loaded_diffusion_cache={cache_path}", flush=True)
            return payload

    dataset = SarTtaDataset(data_root, split, class_names, transform=lambda image: image, view="identity")
    features = []
    labels = []
    samples = []
    for path, label in dataset.samples:
        features.append(diffusion_feature_from_path(path))
        labels.append(label)
        samples.append(str(path))
    payload = {
        "features": torch.from_numpy(np.stack(features)),
        "labels": torch.tensor(labels, dtype=torch.long),
        "paths": samples,
        "class_names": class_names,
        "feature_dim": len(features[0]),
        "weights": "heat_diffusion_stats_v1",
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"saved_diffusion_cache={cache_path}", flush=True)
    return payload


def stack_run_logits(payloads: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    labels = payloads[0]["labels"].long()
    samples = list(payloads[0]["samples"])
    for payload in payloads[1:]:
        if not torch.equal(labels, payload["labels"].long()):
            raise ValueError("Logit label order mismatch")
        if samples != list(payload["samples"]):
            raise ValueError("Logit sample order mismatch")
    logits = torch.cat([payload["logits"].float() for payload in payloads], dim=1)
    return logits, labels, samples


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
    model: ResidualFusion,
    run_logits: torch.Tensor,
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
            logits = model(run_logits[start:end], diffusion_features[start:end])
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

    class_names = class_names_from_train(args.data_root)
    print(f"data_root={args.data_root}", flush=True)
    print(f"class_names={class_names}", flush=True)
    print(f"runs={[str(run) for run in args.runs]}", flush=True)

    train_payloads = [
        compute_or_load_tta_logits(
            run,
            args.data_root,
            "train",
            class_names,
            args.logits_cache_dir,
            args.batch_size,
            args.num_workers,
        )
        for run in args.runs
    ]
    test_payloads = [
        compute_or_load_tta_logits(
            run,
            args.data_root,
            "test",
            class_names,
            args.logits_cache_dir,
            args.batch_size,
            args.num_workers,
        )
        for run in args.runs
    ]
    train_logits, train_labels, train_samples = stack_run_logits(train_payloads)
    test_logits, test_labels, test_samples = stack_run_logits(test_payloads)

    train_diffusion_payload = compute_or_load_diffusion_features(
        args.data_root,
        "train",
        class_names,
        args.diffusion_cache_dir,
    )
    test_diffusion_payload = compute_or_load_diffusion_features(
        args.data_root,
        "test",
        class_names,
        args.diffusion_cache_dir,
    )
    if not torch.equal(train_labels, train_diffusion_payload["labels"].long()):
        raise ValueError("Train diffusion label order mismatch")
    if not torch.equal(test_labels, test_diffusion_payload["labels"].long()):
        raise ValueError("Test diffusion label order mismatch")
    if train_samples != list(train_diffusion_payload["paths"]):
        raise ValueError("Train diffusion sample order mismatch")
    if test_samples != list(test_diffusion_payload["paths"]):
        raise ValueError("Test diffusion sample order mismatch")

    train_diffusion, test_diffusion, diffusion_mean, diffusion_std = normalize_features(
        train_diffusion_payload["features"].float(),
        test_diffusion_payload["features"].float(),
    )

    num_classes = len(class_names)
    model = ResidualFusion(
        num_runs=len(args.runs),
        num_classes=num_classes,
        diffusion_dim=train_diffusion.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        train_logit_branch=args.train_logit_branch,
    )
    base_loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    with torch.no_grad():
        base_test_logits = model.logit_branch(test_logits)
        base_train_logits = model.logit_branch(train_logits)
        base_train_accuracy = accuracy(base_train_logits, train_labels)
        base_test_accuracy = accuracy(base_test_logits, test_labels)
    print(
        f"base_train_accuracy={base_train_accuracy:.4f} "
        f"base_test_accuracy={base_test_accuracy:.4f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    best: dict[str, object] = {
        "test_accuracy": -1.0,
        "epoch": 0,
        "state_dict": None,
    }
    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(train_labels.shape[0])
        for start in range(0, train_labels.shape[0], args.batch_size):
            indices = permutation[start : start + args.batch_size]
            optimizer.zero_grad()
            logits = model(train_logits[indices], train_diffusion[indices])
            loss = loss_fn(logits, train_labels[indices])
            loss.backward()
            optimizer.step()

        train_loss, train_accuracy, _ = evaluate(
            model,
            train_logits,
            train_diffusion,
            train_labels,
            base_loss_fn,
            args.batch_size,
        )
        test_loss, test_accuracy, _ = evaluate(
            model,
            test_logits,
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
                "residual_scale": float(model.residual_scale.detach().cpu()),
                "elapsed_seconds": time.time() - started_at,
            }
        )
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f} "
                f"best={float(best['test_accuracy']):.4f}@{best['epoch']} "
                f"scale={float(model.residual_scale.detach().cpu()):.4f}",
                flush=True,
            )

    assert best["state_dict"] is not None
    model.load_state_dict(best["state_dict"])
    train_loss, train_accuracy, _ = evaluate(
        model,
        train_logits,
        train_diffusion,
        train_labels,
        base_loss_fn,
        args.batch_size,
    )
    test_loss, test_accuracy, test_predictions = evaluate(
        model,
        test_logits,
        test_diffusion,
        test_labels,
        base_loss_fn,
        args.batch_size,
    )
    test_confusion = confusion_matrix(test_labels, test_predictions, num_classes)
    metrics = {
        "data_root": str(args.data_root),
        "runs": [str(run) for run in args.runs],
        "class_names": class_names,
        "num_runs": len(args.runs),
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
        "residual_scale": float(model.residual_scale.detach().cpu()),
        "test_per_class_accuracy": per_class_accuracy(test_confusion, class_names),
        "test_confusion_matrix": test_confusion,
        "elapsed_seconds": time.time() - started_at,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "diffusion_mean": diffusion_mean,
            "diffusion_std": diffusion_std,
            "class_names": class_names,
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
