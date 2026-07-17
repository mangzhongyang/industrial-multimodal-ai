"""Lightweight PyTorch models for industrial edge inference.

Models
------
1. HealthTCN: predicts future health/RUL index from a 5-second PLC history.
2. ResNet34DefectClassifier: ResNet-34 backbone for good/scratch/stain classes.

The module deliberately returns logits for classification. Apply softmax only when
probabilities are needed; use CrossEntropyLoss directly during training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from torchvision.models import ResNet34_Weights, resnet34
except ImportError:  # Keep the TCN usable on minimal CPU-only edge runtimes.
    ResNet34_Weights = None  # type: ignore[assignment,misc]
    resnet34 = None  # type: ignore[assignment]


DEFECT_CLASSES: tuple[str, ...] = ("good", "scratch", "stain")


@dataclass(frozen=True)
class TCNConfig:
    """Settings for a compact causal TCN.

    With a one-Hz PLC stream, ``history_seconds=5`` means model input has shape
    ``[batch, 5, 3]`` (vibration, current, temperature).  ``future_steps=1``
    forecasts the health index one second ahead.
    """

    input_features: int = 3
    channels: tuple[int, ...] = (32, 32, 32)
    kernel_size: int = 3
    dropout: float = 0.10
    future_steps: int = 1
    history_seconds: int = 5
    sample_rate_hz: int = 1


@dataclass(frozen=True)
class VisionConfig:
    num_classes: int = len(DEFECT_CLASSES)
    pretrained: bool = True
    class_names: tuple[str, ...] = DEFECT_CLASSES


class Chomp1d(nn.Module):
    """Removes right padding so a dilated convolution remains causal."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, : -self.size].contiguous() if self.size else x


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.net(x) + self.residual(x))


class HealthTCN(nn.Module):
    """Causal TCN regressor for a normalized health/RUL index in [0, 1]."""

    def __init__(self, config: TCNConfig = TCNConfig()) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        previous = config.input_features
        for level, channels in enumerate(config.channels):
            layers.append(TemporalBlock(previous, channels, config.kernel_size, 2**level, config.dropout))
            previous = channels
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(previous, 16), nn.ReLU(inplace=True), nn.Linear(16, config.future_steps))

    def forward(self, history: Tensor) -> Tensor:
        """Forecast health index.

        Args:
            history: float tensor [batch, time_steps, input_features].
        Returns:
            Tensor [batch, future_steps], clamped by sigmoid to [0, 1].
        """
        if history.ndim != 3:
            raise ValueError("history must have shape [batch, time_steps, input_features]")
        if history.shape[-1] != self.config.input_features:
            raise ValueError(f"Expected {self.config.input_features} features, got {history.shape[-1]}")
        features = self.tcn(history.transpose(1, 2))[:, :, -1]
        return torch.sigmoid(self.head(features))


class ResNet34DefectClassifier(nn.Module):
    """ResNet-34 defect classifier with an inspectable final convolution layer."""

    def __init__(self, config: VisionConfig = VisionConfig()) -> None:
        super().__init__()
        if config.num_classes != len(config.class_names):
            raise ValueError("num_classes must equal the number of class_names")
        if resnet34 is None or ResNet34_Weights is None:
            raise ImportError(
                "ResNet-34 requires torchvision. Install a version compatible with PyTorch: pip install torchvision"
            )
        self.config = config
        weights = ResNet34_Weights.DEFAULT if config.pretrained else None
        self.backbone = resnet34(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, config.num_classes)

    @property
    def cam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1].conv2

    def forward(self, image: Tensor) -> Tensor:
        """Return class logits for ImageNet-normalized image tensor [B, 3, H, W]."""
        return self.backbone(image)

    @torch.inference_mode()
    def predict(self, image: Tensor) -> tuple[Tensor, Tensor, list[str]]:
        logits = self(image)
        probabilities = logits.softmax(dim=1)
        class_ids = probabilities.argmax(dim=1)
        names = [self.config.class_names[index] for index in class_ids.tolist()]
        return class_ids, probabilities, names


class GradCAM:
    """Grad-CAM for ResNet34DefectClassifier.

    Call ``generate`` with a single image (or batch) before converting its output
    to an overlay. The result is a heatmap tensor in [0, 1], shape [B, H, W].
    """

    def __init__(self, model: ResNet34DefectClassifier) -> None:
        self.model = model
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None
        self._forward_handle = model.cam_target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = model.cam_target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self.activations = output

    def _save_gradients(
        self, _module: nn.Module, _grad_inputs: tuple[Tensor | None, ...], grad_outputs: tuple[Tensor, ...]
    ) -> None:
        self.gradients = grad_outputs[0]

    def generate(self, image: Tensor, target_class: int | Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return (heatmaps, selected_class_ids) for an image batch."""
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if target_class is None:
            class_ids = logits.argmax(dim=1)
        elif isinstance(target_class, int):
            class_ids = torch.full((image.size(0),), target_class, dtype=torch.long, device=image.device)
        else:
            class_ids = target_class.to(image.device).long()
        if class_ids.shape != (image.size(0),):
            raise ValueError("target_class tensor must have shape [batch]")
        logits.gather(1, class_ids[:, None]).sum().backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture target-layer values")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        minimum = cam.amin(dim=(1, 2), keepdim=True)
        maximum = cam.amax(dim=(1, 2), keepdim=True)
        heatmaps = (cam - minimum) / (maximum - minimum).clamp_min(1e-8)
        self.model.train(was_training)
        return heatmaps.detach(), class_ids.detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def heatmap_overlay(image: Tensor, heatmap: Tensor, alpha: float = 0.45) -> Tensor:
    """Blend a red/yellow Grad-CAM map over an RGB image tensor in [0, 1]."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if heatmap.ndim == 2:
        heatmap = heatmap.unsqueeze(0)
    if image.ndim != 4 or heatmap.ndim != 3 or image.shape[0] != heatmap.shape[0]:
        raise ValueError("image must be [B,3,H,W] and heatmap must be [B,H,W]")
    heatmap = F.interpolate(heatmap[:, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    color = torch.stack((heatmap, heatmap.square(), 1.0 - heatmap), dim=1)
    return (image.clamp(0, 1) * (1 - alpha) + color * alpha).clamp(0, 1)


ModelKind = Literal["tcn", "resnet34"]


def create_model(kind: ModelKind, **kwargs: Any) -> nn.Module:
    """Create a model; kwargs map to TCNConfig or VisionConfig fields."""
    if kind == "tcn":
        return HealthTCN(TCNConfig(**kwargs))
    if kind == "resnet34":
        return ResNet34DefectClassifier(VisionConfig(**kwargs))
    raise ValueError(f"Unknown model kind: {kind}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Persist weights plus enough model metadata to safely recreate the model."""
    if isinstance(model, HealthTCN):
        kind, config = "tcn", asdict(model.config)
    elif isinstance(model, ResNet34DefectClassifier):
        kind, config = "resnet34", asdict(model.config)
    else:
        raise TypeError("save_checkpoint supports HealthTCN and ResNet34DefectClassifier")
    checkpoint = {
        "model_kind": kind,
        "model_config": config,
        "model_state": model.state_dict(),
        "epoch": epoch,
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load model weights and optional optimizer state; returns (model, metadata)."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = dict(checkpoint["model_config"])
    # A checkpoint already holds all ResNet weights. Never trigger a network
    # download merely because it was originally initialized as pretrained.
    if checkpoint["model_kind"] == "resnet34":
        config["pretrained"] = False
    model = create_model(checkpoint["model_kind"], **config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    metadata = {key: checkpoint.get(key) for key in ("model_kind", "model_config", "epoch", "extra")}
    return model, metadata


if __name__ == "__main__":
    # Smoke-test architecture shapes without downloading pretrained weights.
    tcn = create_model("tcn", pretrained=False) if False else create_model("tcn")
    demo_series = torch.randn(2, 5, 3)
    print("TCN output:", tcn(demo_series).shape)  # [2, 1]
    vision = create_model("resnet34", pretrained=False)
    print("Vision logits:", vision(torch.randn(2, 3, 224, 224)).shape)  # [2, 3]
