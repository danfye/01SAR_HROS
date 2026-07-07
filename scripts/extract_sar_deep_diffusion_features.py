#!/usr/bin/env python3
"""Train a small DDPM denoiser and cache learned SAR diffusion features.

The cached features are activations from a denoising diffusion network trained
with the standard epsilon-prediction objective on the SAR training split. They
are intended for the third branch in ``train_sar_three_branch_fusion.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FEATURE_TYPE = "deep_diffusion_ddpm"
WEIGHTS_NAME = "tiny_sar_ddpm_v1"


@dataclass
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor


class SarImageDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        split: str,
        class_names: list[str],
        image_size: int,
    ) -> None:
        self.image_size = image_size
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        with Image.open(path) as image:
            image = image.convert("L").resize((self.image_size, self.image_size), resampling)
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        tensor = tensor.mul(2.0).sub(1.0)
        return tensor, label, str(path)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        denominator = max(half - 1, 1)
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / denominator
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.shape[1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[1]))
        return embedding


class TimeResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(norm_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(inputs)))
        hidden = hidden + self.time_proj(F.silu(time_embedding)).unsqueeze(-1).unsqueeze(-1)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class TinyDiffusionUNet(nn.Module):
    def __init__(self, base_channels: int, time_dim: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.enc1 = TimeResBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1)
        self.enc2 = TimeResBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1)
        self.enc3 = TimeResBlock(c3, c3, time_dim)
        self.middle = TimeResBlock(c3, c3, time_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec2 = TimeResBlock(c2 + c2, c2, time_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = TimeResBlock(c1 + c1, c1, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(norm_groups(c1), c1),
            nn.SiLU(),
            nn.Conv2d(c1, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        timesteps: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        time_embedding = self.time_mlp(self.time_embedding(timesteps))
        stem = self.stem(inputs)
        enc1 = self.enc1(stem, time_embedding)
        enc2 = self.enc2(self.down1(enc1), time_embedding)
        enc3 = self.enc3(self.down2(enc2), time_embedding)
        middle = self.middle(enc3, time_embedding)
        dec2 = self.dec2(torch.cat([self.up2(middle), enc2], dim=1), time_embedding)
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1), time_embedding)
        prediction = self.out(dec1)
        if return_features:
            return prediction, [enc1, enc2, middle]
        return prediction


def norm_groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DDPM-style SAR denoiser and cache learned diffusion features."
    )
    parser.add_argument("--data-root", default=Path("SAR"), type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("results/sar_finetune/deep_diffusion_features"),
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="Optional denoiser checkpoint. Defaults to <output-dir>/diffusion_encoder.pt.",
    )
    parser.add_argument("--image-size", default=64, type=int)
    parser.add_argument("--timesteps", default=200, type=int)
    parser.add_argument(
        "--feature-timesteps",
        default="0,50,100,150",
        help="Comma-separated diffusion timesteps used when extracting activations.",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--feature-batch-size", default=128, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--time-dim", default=128, type=int)
    parser.add_argument("--gradient-clip", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--threads", default=8, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even when the checkpoint already exists.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Skip training and require an existing checkpoint.",
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
    class_names = [
        path.name
        for path in sorted(train_dir.iterdir(), key=lambda path: path.name.casefold())
        if path.is_dir()
    ]
    if not class_names:
        raise ValueError(f"No class directories found in {train_dir}")
    return class_names


def parse_timestep_list(value: str, num_timesteps: int) -> list[int]:
    timesteps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not timesteps:
        raise ValueError("--feature-timesteps must contain at least one timestep")
    for timestep in timesteps:
        if timestep < 0 or timestep >= num_timesteps:
            raise ValueError(
                f"Feature timestep {timestep} is outside [0, {num_timesteps - 1}]"
            )
    return timesteps


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_schedule(num_timesteps: int, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
    )


def q_sample(
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    shape = (clean_images.shape[0], 1, 1, 1)
    sqrt_alpha = schedule.sqrt_alpha_bars[timesteps].view(shape)
    sqrt_one_minus_alpha = schedule.sqrt_one_minus_alpha_bars[timesteps].view(shape)
    return sqrt_alpha * clean_images + sqrt_one_minus_alpha * noise


def train_denoiser(
    model: TinyDiffusionUNet,
    train_dataset: SarImageDataset,
    args: argparse.Namespace,
    device: torch.device,
    schedule: DiffusionSchedule,
) -> list[dict[str, float]]:
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history: list[dict[str, float]] = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for images, _labels, _paths in loader:
            images = images.to(device)
            timesteps = torch.randint(
                0,
                args.timesteps,
                (images.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(images)
            noisy_images = q_sample(images, timesteps, noise, schedule)
            prediction = model(noisy_images, timesteps)
            assert isinstance(prediction, torch.Tensor)
            loss = F.mse_loss(prediction, noise)

            optimizer.zero_grad()
            loss.backward()
            if args.gradient_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()

            total_loss += float(loss.item()) * images.shape[0]
            total_count += images.shape[0]

        average_loss = total_loss / total_count
        history.append(
            {
                "epoch": float(epoch),
                "train_mse": average_loss,
                "elapsed_seconds": time.time() - started_at,
            }
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d}/{args.epochs} train_mse={average_loss:.6f}",
                flush=True,
            )
    return history


def activation_features(activations: list[torch.Tensor]) -> torch.Tensor:
    blocks = []
    for activation in activations:
        pooled = F.adaptive_avg_pool2d(activation, (2, 2)).flatten(1)
        mean = activation.mean(dim=(2, 3))
        std = activation.std(dim=(2, 3), unbiased=False)
        blocks.extend([pooled, mean, std])
    return torch.cat(blocks, dim=1)


def extract_features(
    model: TinyDiffusionUNet,
    dataset: SarImageDataset,
    args: argparse.Namespace,
    device: torch.device,
    schedule: DiffusionSchedule,
    feature_timesteps: list[int],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    loader = DataLoader(
        dataset,
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    paths: list[str] = []
    model.eval()
    with torch.no_grad():
        for batch_index, (images, targets, batch_paths) in enumerate(loader):
            images = images.to(device)
            timestep_features = []
            for timestep in feature_timesteps:
                timesteps = torch.full(
                    (images.shape[0],),
                    timestep,
                    dtype=torch.long,
                    device=device,
                )
                generator = torch.Generator(device=device)
                generator.manual_seed(args.seed + 1729 + timestep * 1000003 + batch_index)
                noise = torch.randn(
                    images.shape,
                    generator=generator,
                    device=device,
                    dtype=images.dtype,
                )
                noisy_images = q_sample(images, timesteps, noise, schedule)
                prediction, activations = model(noisy_images, timesteps, return_features=True)
                assert isinstance(prediction, torch.Tensor)
                timestep_features.append(activation_features(activations).cpu())
            features.append(torch.cat(timestep_features, dim=1))
            labels.append(targets.long())
            paths.extend(str(path) for path in batch_paths)
    return torch.cat(features), torch.cat(labels), paths


def feature_cache_path(output_dir: Path, data_root: Path, split: str) -> Path:
    return output_dir / f"{data_root.name}_{split}_deep_diffusion.pt"


def save_feature_cache(
    output_dir: Path,
    data_root: Path,
    split: str,
    features: torch.Tensor,
    labels: torch.Tensor,
    paths: list[str],
    class_names: list[str],
    feature_timesteps: list[int],
    checkpoint_path: Path,
    args: argparse.Namespace,
) -> Path:
    path = feature_cache_path(output_dir, data_root, split)
    payload = {
        "features": features,
        "labels": labels,
        "paths": paths,
        "class_names": class_names,
        "feature_dim": int(features.shape[1]),
        "feature_type": FEATURE_TYPE,
        "weights": WEIGHTS_NAME,
        "checkpoint": str(checkpoint_path),
        "feature_timesteps": feature_timesteps,
        "image_size": args.image_size,
        "timesteps": args.timesteps,
        "base_channels": args.base_channels,
        "time_dim": args.time_dim,
        "trained_on": f"{data_root}/train",
    }
    torch.save(payload, path)
    print(f"saved_feature_cache={path}", flush=True)
    return path


def save_checkpoint(
    model: TinyDiffusionUNet,
    checkpoint_path: Path,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "class_names": class_names,
        "feature_type": FEATURE_TYPE,
        "weights": WEIGHTS_NAME,
        "args": serializable_args(args),
    }
    torch.save(payload, checkpoint_path)
    print(f"saved_checkpoint={checkpoint_path}", flush=True)


def load_checkpoint(model: TinyDiffusionUNet, checkpoint_path: Path) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    print(f"loaded_checkpoint={checkpoint_path}", flush=True)
    return payload


def write_history(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> None:
    args = parse_args()
    if args.image_size % 4 != 0:
        raise ValueError("--image-size must be divisible by 4 for this U-Net")
    if args.epochs < 1 and not args.extract_only:
        raise ValueError("--epochs must be at least 1 unless --extract-only is used")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.output_dir / "diffusion_encoder.pt"
    feature_timesteps = parse_timestep_list(args.feature_timesteps, args.timesteps)
    device = resolve_device(args.device)
    schedule = make_schedule(args.timesteps, device)

    class_names = class_names_from_train(args.data_root)
    train_dataset = SarImageDataset(args.data_root, "train", class_names, args.image_size)
    test_dataset = SarImageDataset(args.data_root, "test", class_names, args.image_size)
    model = TinyDiffusionUNet(args.base_channels, args.time_dim).to(device)

    history: list[dict[str, float]] = []
    if checkpoint_path.exists() and not args.force_train:
        load_checkpoint(model, checkpoint_path)
        model.to(device)
    else:
        if args.extract_only:
            raise FileNotFoundError(f"Missing checkpoint for --extract-only: {checkpoint_path}")
        print(
            f"training_diffusion_denoiser samples={len(train_dataset)} "
            f"device={device} timesteps={args.timesteps}",
            flush=True,
        )
        history = train_denoiser(model, train_dataset, args, device, schedule)
        save_checkpoint(model, checkpoint_path, class_names, args)
        model.to(device)

    print(f"extracting_features split=train feature_timesteps={feature_timesteps}", flush=True)
    train_features, train_labels, train_paths = extract_features(
        model,
        train_dataset,
        args,
        device,
        schedule,
        feature_timesteps,
    )
    print(f"extracting_features split=test feature_timesteps={feature_timesteps}", flush=True)
    test_features, test_labels, test_paths = extract_features(
        model,
        test_dataset,
        args,
        device,
        schedule,
        feature_timesteps,
    )

    train_cache = save_feature_cache(
        args.output_dir,
        args.data_root,
        "train",
        train_features,
        train_labels,
        train_paths,
        class_names,
        feature_timesteps,
        checkpoint_path,
        args,
    )
    test_cache = save_feature_cache(
        args.output_dir,
        args.data_root,
        "test",
        test_features,
        test_labels,
        test_paths,
        class_names,
        feature_timesteps,
        checkpoint_path,
        args,
    )

    write_history(args.output_dir / "diffusion_training_history.csv", history)
    metrics = {
        "feature_type": FEATURE_TYPE,
        "weights": WEIGHTS_NAME,
        "data_root": str(args.data_root),
        "class_names": class_names,
        "checkpoint": str(checkpoint_path),
        "train_cache": str(train_cache),
        "test_cache": str(test_cache),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "feature_dim": int(train_features.shape[1]),
        "feature_timesteps": feature_timesteps,
        "args": serializable_args(args),
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(f"feature_dim={train_features.shape[1]}", flush=True)
    print(f"saved_metrics={args.output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
