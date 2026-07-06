#!/usr/bin/env python3
"""Train a minimal image classifier with only NumPy and Pillow.

Expected data layout:

    RGB/
      train/<class_name>/*.png
      test/<class_name>/*.png

The model is multinomial logistic regression over resized raw pixels. It is
intentionally simple so it can run in this repository without PyTorch or
scikit-learn.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a basic NumPy softmax image classifier."
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
        help="Directory for model and metrics. Defaults to results/basic_image_classifier/<data-root>.",
    )
    parser.add_argument(
        "--image-size",
        default=32,
        type=int,
        help="Resize images to image-size x image-size before training.",
    )
    parser.add_argument("--epochs", default=80, type=int, help="Training epochs.")
    parser.add_argument("--batch-size", default=128, type=int, help="Mini-batch size.")
    parser.add_argument("--learning-rate", default=0.05, type=float, help="SGD learning rate.")
    parser.add_argument("--l2", default=1e-4, type=float, help="L2 regularization strength.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "--limit-per-class",
        default=None,
        type=int,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def class_directories(split_dir: Path) -> list[Path]:
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    classes = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class directories found under {split_dir}")
    return classes


def image_paths_for_class(class_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No image files found under {class_dir}")
    return paths


def resolve_split_class_dir(split_dir: Path, class_name: str) -> Path:
    exact = split_dir / class_name
    if exact.exists():
        return exact

    by_casefold = {
        path.name.casefold(): path for path in split_dir.iterdir() if path.is_dir()
    }
    resolved = by_casefold.get(class_name.casefold())
    if resolved is None:
        raise FileNotFoundError(f"Missing class {class_name!r} under {split_dir}")
    return resolved


def load_split(
    split_dir: Path,
    class_names: list[str],
    image_size: int,
    limit_per_class: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    counts: dict[str, int] = {}

    resampling = getattr(Image, "Resampling", Image).BILINEAR
    for class_index, class_name in enumerate(class_names):
        class_dir = resolve_split_class_dir(split_dir, class_name)
        paths = image_paths_for_class(class_dir, limit_per_class)
        counts[class_name] = len(paths)

        for path in paths:
            with Image.open(path) as image:
                image = image.convert("RGB").resize((image_size, image_size), resampling)
                array = np.asarray(image, dtype=np.float32) / 255.0
            xs.append(array.reshape(-1))
            ys.append(class_index)

    return np.stack(xs), np.asarray(ys, dtype=np.int64), counts


def standardize(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def evaluate(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    l2: float,
) -> tuple[float, float, np.ndarray]:
    probabilities = softmax(x @ weights + bias)
    n = y.shape[0]
    loss = -np.log(probabilities[np.arange(n), y] + 1e-12).mean()
    loss += 0.5 * l2 * float(np.sum(weights * weights))
    predictions = probabilities.argmax(axis=1)
    accuracy = float((predictions == y).mean())
    return float(loss), accuracy, predictions


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(args.seed)
    num_samples, num_features = x_train.shape

    weights = rng.normal(0.0, 0.01, size=(num_features, num_classes)).astype(np.float32)
    bias = np.zeros((num_classes,), dtype=np.float32)
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        indices = rng.permutation(num_samples)
        for start in range(0, num_samples, args.batch_size):
            batch_indices = indices[start : start + args.batch_size]
            xb = x_train[batch_indices]
            yb = y_train[batch_indices]

            probabilities = softmax(xb @ weights + bias)
            probabilities[np.arange(yb.shape[0]), yb] -= 1.0
            probabilities /= yb.shape[0]

            grad_w = xb.T @ probabilities + args.l2 * weights
            grad_b = probabilities.sum(axis=0)

            weights -= args.learning_rate * grad_w.astype(np.float32)
            bias -= args.learning_rate * grad_b.astype(np.float32)

        train_loss, train_accuracy, _ = evaluate(
            x_train, y_train, weights, bias, args.l2
        )
        test_loss, test_accuracy, _ = evaluate(x_test, y_test, weights, bias, args.l2)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}",
                flush=True,
            )

    return weights, bias, history


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(y_true, y_pred):
        matrix[int(target), int(prediction)] += 1
    return matrix


def per_class_accuracy(
    matrix: np.ndarray, class_names: list[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        total = int(matrix[index].sum())
        result[class_name] = float(matrix[index, index] / total) if total else 0.0
    return result


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


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("results") / "basic_image_classifier" / data_root.name
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    class_names = [path.name for path in class_directories(train_dir)]

    print(f"data_root={data_root}")
    print(f"classes={class_names}")
    print("loading images...")
    x_train, y_train, train_counts = load_split(
        train_dir, class_names, args.image_size, args.limit_per_class
    )
    x_test, y_test, test_counts = load_split(
        test_dir, class_names, args.image_size, args.limit_per_class
    )
    print(f"train_shape={x_train.shape} test_shape={x_test.shape}")

    x_train, x_test, mean, std = standardize(x_train, x_test)
    weights, bias, history = train(
        x_train,
        y_train,
        x_test,
        y_test,
        num_classes=len(class_names),
        args=args,
    )

    train_loss, train_accuracy, train_predictions = evaluate(
        x_train, y_train, weights, bias, args.l2
    )
    test_loss, test_accuracy, test_predictions = evaluate(
        x_test, y_test, weights, bias, args.l2
    )
    train_confusion = confusion_matrix(y_train, train_predictions, len(class_names))
    test_confusion = confusion_matrix(y_test, test_predictions, len(class_names))

    metrics = {
        "data_root": str(data_root),
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "l2": args.l2,
        "seed": args.seed,
        "class_names": class_names,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "train_per_class_accuracy": per_class_accuracy(train_confusion, class_names),
        "test_per_class_accuracy": per_class_accuracy(test_confusion, class_names),
        "train_confusion_matrix": train_confusion.tolist(),
        "test_confusion_matrix": test_confusion.tolist(),
        "elapsed_seconds": time.time() - started_at,
    }

    np.savez_compressed(
        output_dir / "model.npz",
        weights=weights,
        bias=bias,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        class_names=np.asarray(class_names),
        image_size=np.asarray([args.image_size], dtype=np.int64),
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    write_history(output_dir / "history.csv", history)

    print(f"saved_model={output_dir / 'model.npz'}")
    print(f"saved_metrics={output_dir / 'metrics.json'}")
    print(f"final_train_accuracy={train_accuracy:.4f}")
    print(f"final_test_accuracy={test_accuracy:.4f}")


if __name__ == "__main__":
    main()
