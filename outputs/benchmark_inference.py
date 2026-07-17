"""Compare native PyTorch CPU and ONNX Runtime CPU inference performance.

Example:
    python benchmark_inference.py --checkpoint models/tcn.pt --onnx models/tcn.onnx --output benchmark.png

For a fair Intel CPU comparison, the script pins both engines to the CPU and
reports average latency plus end-to-end throughput at the requested batch size.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Callable

# Keep Matplotlib's cache in a writable project-local location on restricted
# Windows/edge accounts before importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from export_onnx import create_onnx_session, dummy_input, run_onnx
from model_factory import load_checkpoint


def measure(inference: Callable[[], object], warmup: int, iterations: int, batch_size: int) -> tuple[float, float]:
    """Return average per-batch latency (ms) and sample throughput (samples/s)."""
    for _ in range(warmup):
        inference()
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        inference()
        durations.append(time.perf_counter() - start)
    mean_seconds = float(np.mean(durations))
    return mean_seconds * 1_000, batch_size / mean_seconds


def plot_results(results: dict[str, tuple[float, float]], output: str | Path) -> Path:
    """Save a two-panel latency/throughput bar chart."""
    labels = list(results)
    latency = [results[name][0] for name in labels]
    throughput = [results[name][1] for name in labels]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for axis, values, title, unit, color in (
        (axes[0], latency, "Latency", "ms / batch", "#457b9d"),
        (axes[1], throughput, "Throughput", "samples / second", "#2a9d8f"),
    ):
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.2f", padding=3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="PyTorch .pt checkpoint")
    parser.add_argument("--onnx", required=True, help="matching ONNX model")
    parser.add_argument("--output", default="benchmark.png", help="destination bar-chart image")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU threads for reproducible comparison")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("batch-size and iterations must be positive; warmup cannot be negative")
    torch.set_num_threads(args.torch_threads)
    model, metadata = load_checkpoint(args.checkpoint, device="cpu")
    model.eval()
    inputs = dummy_input(model, args.batch_size)
    session = create_onnx_session(args.onnx, provider="cpu")

    with torch.inference_mode():
        torch_latency, torch_throughput = measure(lambda: model(inputs), args.warmup, args.iterations, args.batch_size)
    onnx_latency, onnx_throughput = measure(
        lambda: run_onnx(session, inputs), args.warmup, args.iterations, args.batch_size
    )
    results = {"PyTorch CPU": (torch_latency, torch_throughput), "ONNX Runtime CPU": (onnx_latency, onnx_throughput)}
    chart = plot_results(results, args.output)
    print(f"Model: {metadata['model_kind']}")
    for engine, (latency, throughput) in results.items():
        print(f"{engine}: latency={latency:.3f} ms/batch, throughput={throughput:.2f} samples/s")
    print(f"Chart saved to: {chart.resolve()}")
