"""Export an industrial PyTorch checkpoint to ONNX and run it with ONNX Runtime.

Examples:
    python export_onnx.py --checkpoint models/tcn.pt --output models/tcn.onnx
    python export_onnx.py --checkpoint models/vision.pt --output models/vision.onnx --provider cuda

Install: pip install onnx onnxruntime   # CPU
         pip install onnxruntime-gpu     # NVIDIA CUDA GPU
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from model_factory import HealthTCN, ResNet34DefectClassifier, load_checkpoint

Provider = Literal["cpu", "cuda"]


def dummy_input(model: torch.nn.Module, batch_size: int = 1) -> torch.Tensor:
    """Return the correct representative input for either supported model."""
    if isinstance(model, HealthTCN):
        length = model.config.history_seconds * model.config.sample_rate_hz
        return torch.randn(batch_size, length, model.config.input_features)
    if isinstance(model, ResNet34DefectClassifier):
        return torch.randn(batch_size, 3, 224, 224)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def export_model(checkpoint: str | Path, output: str | Path, opset: int = 17) -> Path:
    """Load an application checkpoint and export a batch-dynamic ONNX graph."""
    model, metadata = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    example = dummy_input(model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        output,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported {metadata['model_kind']} to {output.resolve()}")
    return output


def create_onnx_session(onnx_path: str | Path, provider: Provider = "cpu"):
    """Create CPU or CUDA ONNX Runtime session with a clear availability error."""
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("ONNX Runtime is required: pip install onnxruntime (or onnxruntime-gpu)") from error
    requested = "CUDAExecutionProvider" if provider == "cuda" else "CPUExecutionProvider"
    available = ort.get_available_providers()
    if requested not in available:
        raise RuntimeError(f"{requested} is unavailable. Installed providers: {available}")
    return ort.InferenceSession(str(onnx_path), providers=[requested])


def run_onnx(session, input_tensor: torch.Tensor) -> np.ndarray:
    """Run ONNX Runtime inference; tensor is moved to CPU NumPy automatically."""
    input_name = session.get_inputs()[0].name
    return session.run(None, {input_name: input_tensor.detach().cpu().numpy()})[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help=".pt checkpoint created by save_checkpoint")
    parser.add_argument("--output", required=True, help="destination .onnx file")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu", help="run a post-export sanity inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exported_path = export_model(args.checkpoint, args.output, args.opset)
    model, _ = load_checkpoint(args.checkpoint, device="cpu")
    session = create_onnx_session(exported_path, args.provider)
    result = run_onnx(session, dummy_input(model))
    print(f"ONNX Runtime ({args.provider}) output shape: {result.shape}")
