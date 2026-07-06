#!/usr/bin/env python3
"""Plot the SAR test confusion matrix from saved ensemble metrics.

The project already stores the numeric confusion matrix in
`results/sar_finetune/tta_ensemble_metrics.json`. This script exports a CSV and
a readable PNG/SVG figure so the per-class behavior can be inspected directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAR confusion matrix.")
    parser.add_argument(
        "--metrics",
        default=Path("results/sar_finetune/tta_ensemble_metrics.json"),
        type=Path,
    )
    parser.add_argument(
        "--out-base",
        default=Path("image/sar_tta_ensemble_confusion_matrix"),
        type=Path,
        help="Output base path without extension.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[list[str], list[list[int]], float]:
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    return (
        list(metrics["class_names"]),
        [list(map(int, row)) for row in metrics["confusion_matrix"]],
        float(metrics["accuracy"]),
    )


def write_csv(path: Path, class_names: list[str], matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + row)


def cell_color(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    # White to blue.
    r = int(245 - value * 200)
    g = int(248 - value * 150)
    b = int(255 - value * 35)
    return r, g, b


def text_color(value: float) -> tuple[int, int, int]:
    return (255, 255, 255) if value >= 0.52 else (30, 42, 56)


def safe_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = box
    bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = left + (right - left - width) / 2
    y = top + (bottom - top - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, spacing=2, align="center")


def draw_rotated_label(
    image: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    probe = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
    probe_draw = ImageDraw.Draw(probe)
    bbox = probe_draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    temp = Image.new("RGBA", (text_width + 18, text_height + 12), (255, 255, 255, 0))
    draw = ImageDraw.Draw(temp)
    draw.text((9, 4), text, font=font, fill=fill + (255,))
    temp = temp.rotate(45, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(temp, (int(center[0] - temp.width / 2), int(center[1] - temp.height / 2)))


def plot_png(
    path: Path,
    class_names: list[str],
    matrix: list[list[int]],
    accuracy: float,
) -> None:
    n = len(class_names)
    cell = 78
    left = 165
    top = 225
    right_pad = 52
    bottom_pad = 100
    width = left + n * cell + right_pad
    height = top + n * cell + bottom_pad
    image = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)

    title_font = safe_font(24, bold=True)
    subtitle_font = safe_font(15)
    label_font = safe_font(13, bold=True)
    axis_font = safe_font(15, bold=True)
    cell_font = safe_font(13, bold=True)

    draw.text((24, 22), "SAR Test Confusion Matrix", font=title_font, fill=(22, 28, 36))
    draw.text(
        (24, 58),
        f"Equal-weight TTA ensemble, n={sum(sum(row) for row in matrix)}, accuracy={accuracy:.4f}",
        font=subtitle_font,
        fill=(72, 80, 92),
    )
    draw.text((left + n * cell / 2 - 55, 105), "Predicted class", font=axis_font, fill=(30, 42, 56))
    draw.text((22, top + n * cell / 2 - 10), "True class", font=axis_font, fill=(30, 42, 56))

    row_totals = [sum(row) for row in matrix]
    for index, class_name in enumerate(class_names):
        x = left + index * cell
        draw_rotated_label(
            image,
            (x + cell // 2, top - 55),
            class_name,
            label_font,
            (30, 42, 56),
        )
        y = top + index * cell
        draw.text((18, y + cell / 2 - 8), class_name, font=label_font, fill=(30, 42, 56))

    for row_index, row in enumerate(matrix):
        total = max(row_totals[row_index], 1)
        for col_index, count in enumerate(row):
            value = count / total
            x0 = left + col_index * cell
            y0 = top + row_index * cell
            x1 = x0 + cell
            y1 = y0 + cell
            draw.rectangle([x0, y0, x1, y1], fill=cell_color(value), outline=(215, 223, 232))
            draw_centered(
                draw,
                (x0, y0, x1, y1),
                f"{count}\n{value * 100:.0f}%",
                cell_font,
                text_color(value),
            )

    # Border around matrix.
    draw.rectangle(
        [left, top, left + n * cell, top + n * cell],
        outline=(52, 73, 94),
        width=2,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, quality=95)


def plot_svg(
    path: Path,
    class_names: list[str],
    matrix: list[list[int]],
    accuracy: float,
) -> None:
    n = len(class_names)
    cell = 78
    left = 165
    top = 225
    width = left + n * cell + 52
    height = top + n * cell + 100
    row_totals = [sum(row) for row in matrix]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,DejaVu Sans,sans-serif}.title{font-size:24px;font-weight:700}.sub{font-size:15px;fill:#48505c}.label{font-size:13px;font-weight:700;fill:#1e2a38}.axis{font-size:15px;font-weight:700;fill:#1e2a38}.cell{font-size:13px;font-weight:700;text-anchor:middle}</style>',
        '<text x="24" y="42" class="title" fill="#161c24">SAR Test Confusion Matrix</text>',
        f'<text x="24" y="72" class="sub">Equal-weight TTA ensemble, n={sum(sum(row) for row in matrix)}, accuracy={accuracy:.4f}</text>',
        f'<text x="{left + n * cell / 2 - 55:.0f}" y="105" class="axis">Predicted class</text>',
        f'<text x="22" y="{top + n * cell / 2:.0f}" class="axis">True class</text>',
    ]
    for index, class_name in enumerate(class_names):
        x = left + index * cell + cell / 2
        y = top - 55
        lines.append(f'<text x="{x:.1f}" y="{y:.1f}" class="label" transform="rotate(-45 {x:.1f} {y:.1f})" text-anchor="middle">{escape(class_name)}</text>')
        lines.append(f'<text x="18" y="{top + index * cell + cell / 2 + 5:.1f}" class="label">{escape(class_name)}</text>')
    for row_index, row in enumerate(matrix):
        total = max(row_totals[row_index], 1)
        for col_index, count in enumerate(row):
            value = count / total
            r, g, b = cell_color(value)
            fill = f"rgb({r},{g},{b})"
            text_fill = "white" if value >= 0.52 else "#1e2a38"
            x = left + col_index * cell
            y = top + row_index * cell
            cx = x + cell / 2
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#d7dfe8"/>')
            lines.append(f'<text x="{cx}" y="{y + 34}" class="cell" fill="{text_fill}">{count}</text>')
            lines.append(f'<text x="{cx}" y="{y + 52}" class="cell" fill="{text_fill}">{value * 100:.0f}%</text>')
    lines.append(f'<rect x="{left}" y="{top}" width="{n * cell}" height="{n * cell}" fill="none" stroke="#34495e" stroke-width="2"/>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    class_names, matrix, accuracy = load_metrics(args.metrics)
    write_csv(args.out_base.with_suffix(".csv"), class_names, matrix)
    plot_png(args.out_base.with_suffix(".png"), class_names, matrix, accuracy)
    plot_svg(args.out_base.with_suffix(".svg"), class_names, matrix, accuracy)
    print(f"saved_csv={args.out_base.with_suffix('.csv')}")
    print(f"saved_png={args.out_base.with_suffix('.png')}")
    print(f"saved_svg={args.out_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
