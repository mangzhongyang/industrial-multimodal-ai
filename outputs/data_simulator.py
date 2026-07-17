#!/usr/bin/env python3
"""Reproducible industrial multimodal data simulator.

Generates correlated PLC time-series measurements and product-surface images.
Image annotations use YOLO detection format:
    class_id x_center y_center width height
where all coordinates are normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASS_NAMES = {0: "scratch", 1: "stain"}


@dataclass(frozen=True)
class Config:
    output_dir: str = "synthetic_factory_data"
    num_samples: int = 300
    image_width: int = 640
    image_height: int = 480
    sample_interval_s: float = 1.0
    seed: int = 20250718


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def prepare_output(root: Path) -> tuple[Path, Path]:
    """Recreate generated folders so repeated runs cannot retain stale samples."""
    images_dir = root / "images"
    labels_dir = root / "labels"
    root.mkdir(parents=True, exist_ok=True)
    for folder in (images_dir, labels_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    return images_dir, labels_dir


def degradation_curve(index: int, count: int) -> float:
    """Nonlinear bearing-wear curve, normalized to [0, 1]."""
    progress = index / max(count - 1, 1)
    # Slow early wear followed by accelerating degradation.
    return progress**2.2


def generate_plc_row(
    index: int,
    config: Config,
    np_rng: np.random.Generator,
) -> dict[str, int | float | str]:
    d = degradation_curve(index, config.num_samples)
    t = index * config.sample_interval_s

    # Periodic operating-load changes plus stochastic sensor noise.
    load = 0.55 + 0.12 * math.sin(2 * math.pi * t / 75.0)
    vibration = 1.05 + 0.45 * load + 5.8 * d**1.35
    vibration += 0.12 * math.sin(2 * math.pi * t / 9.0) + np_rng.normal(0, 0.10 + 0.18 * d)

    current = 8.8 + 4.2 * load + 3.2 * d
    current += 0.35 * math.sin(2 * math.pi * t / 20.0) + np_rng.normal(0, 0.18 + 0.12 * d)

    temperature = 38.0 + 9.0 * load + 25.0 * d**1.15
    temperature += 0.8 * math.sin(2 * math.pi * t / 120.0) + np_rng.normal(0, 0.35)

    if d < 0.35:
        state = "normal"
    elif d < 0.70:
        state = "warning"
    else:
        state = "degraded"

    return {
        "sample_id": f"sample_{index:06d}",
        "timestamp_s": round(t, 3),
        "vibration_mm_s": round(max(0.0, vibration), 4),
        "current_a": round(max(0.0, current), 4),
        "temperature_c": round(temperature, 4),
        "degradation": round(d, 6),
        "machine_state": state,
    }


def base_surface(width: int, height: int, np_rng: np.random.Generator) -> Image.Image:
    """Create a lightly textured metallic product surface."""
    x_gradient = np.linspace(0, 12, width, dtype=np.float32)[None, :]
    noise = np_rng.normal(0, 3.0, (height, width)).astype(np.float32)
    gray = np.clip(184 + x_gradient + noise, 0, 255).astype(np.uint8)
    rgb = np.dstack((gray + 7, gray + 4, gray))
    image = Image.fromarray(rgb.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for y in range(25, height, 38):
        draw.line((0, y, width, y), fill=(205, 207, 205), width=1)
    return image


def draw_scratch(
    image: Image.Image,
    rng: random.Random,
) -> tuple[int, float, float, float, float]:
    width, height = image.size
    length = rng.randint(max(24, width // 16), max(30, width // 4))
    thickness = rng.randint(2, 6)
    angle = rng.uniform(-math.pi, math.pi)
    cx = rng.randint(length // 2 + 3, width - length // 2 - 4)
    cy = rng.randint(length // 2 + 3, height - length // 2 - 4)
    dx, dy = math.cos(angle) * length / 2, math.sin(angle) * length / 2
    x1, y1, x2, y2 = cx - dx, cy - dy, cx + dx, cy + dy
    draw = ImageDraw.Draw(image)
    draw.line((x1, y1, x2, y2), fill=(70, 72, 68), width=thickness)
    draw.line((x1 + 1, y1 - 1, x2 + 1, y2 - 1), fill=(225, 225, 218), width=1)
    pad = thickness + 3
    return 0, min(x1, x2) - pad, min(y1, y2) - pad, max(x1, x2) + pad, max(y1, y2) + pad


def draw_stain(
    image: Image.Image,
    rng: random.Random,
) -> tuple[int, float, float, float, float]:
    width, height = image.size
    rx = rng.randint(max(10, width // 45), max(18, width // 12))
    ry = rng.randint(max(8, height // 50), max(15, height // 10))
    cx = rng.randint(rx + 2, width - rx - 3)
    cy = rng.randint(ry + 2, height - ry - 3)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    points = []
    for step in range(18):
        angle = 2 * math.pi * step / 18
        scale = rng.uniform(0.72, 1.18)
        points.append((cx + math.cos(angle) * rx * scale, cy + math.sin(angle) * ry * scale))
    draw.polygon(points, fill=(88, 67, 38, rng.randint(90, 150)))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2.2))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    pad = 5
    return 1, cx - rx - pad, cy - ry - pad, cx + rx + pad, cy + ry + pad


def to_yolo(box: tuple[int, float, float, float, float], width: int, height: int) -> str:
    class_id, x1, y1, x2, y2 = box
    x1, x2 = clamp(x1, 0, width - 1), clamp(x2, 0, width - 1)
    y1, y2 = clamp(y1, 0, height - 1), clamp(y2, 0, height - 1)
    xc = ((x1 + x2) / 2) / width
    yc = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def generate_image_and_labels(
    degradation: float,
    config: Config,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[Image.Image, list[str]]:
    image = base_surface(config.image_width, config.image_height, np_rng)
    boxes: list[tuple[int, float, float, float, float]] = []

    # Defect probability/count grows with bearing degradation, preserving some
    # clean and imperfect examples at every stage.
    scratch_count = sum(rng.random() < (0.10 + 0.30 * degradation) for _ in range(3))
    stain_count = sum(rng.random() < (0.07 + 0.25 * degradation) for _ in range(2))
    for _ in range(scratch_count):
        boxes.append(draw_scratch(image, rng))
    for _ in range(stain_count):
        boxes.append(draw_stain(image, rng))

    labels = [to_yolo(box, config.image_width, config.image_height) for box in boxes]
    return image, labels


def simulate(config: Config) -> Path:
    if config.num_samples <= 0:
        raise ValueError("num_samples must be greater than zero")
    if config.image_width < 128 or config.image_height < 128:
        raise ValueError("image dimensions must both be at least 128 pixels")

    rng = random.Random(config.seed)
    np_rng = np.random.default_rng(config.seed)
    root = Path(config.output_dir).expanduser().resolve()
    images_dir, labels_dir = prepare_output(root)

    rows = []
    for index in range(config.num_samples):
        row = generate_plc_row(index, config, np_rng)
        sample_id = str(row["sample_id"])
        image, labels = generate_image_and_labels(float(row["degradation"]), config, rng, np_rng)
        image.save(images_dir / f"{sample_id}.png", optimize=False)
        (labels_dir / f"{sample_id}.txt").write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        row["image_path"] = f"images/{sample_id}.png"
        row["label_path"] = f"labels/{sample_id}.txt"
        row["defect_count"] = len(labels)
        rows.append(row)

    csv_path = root / "plc_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (root / "classes.txt").write_text("\n".join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)) + "\n", encoding="utf-8")
    metadata = {
        "config": asdict(config),
        "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "yolo_format": "class_id x_center y_center width height (normalized)",
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="synthetic_factory_data")
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20250718)
    args = parser.parse_args()
    return Config(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        image_width=args.width,
        image_height=args.height,
        sample_interval_s=args.sample_interval,
        seed=args.seed,
    )


if __name__ == "__main__":
    output = simulate(parse_args())
    print(f"Synthetic multimodal dataset written to: {output}")
