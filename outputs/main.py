"""FastAPI edge-inference API for industrial multimodal prediction.

Run from this directory:
    pip install fastapi "uvicorn[standard]" redis python-multipart torch pillow numpy torchvision
    uvicorn main:app --host 0.0.0.0 --port 8000

The API accepts multipart/form-data because it carries both the product image and
the JSON sensor history. `sensor_data` is a JSON string of 5 (or more) readings:
[{"timestamp_s": 0, "vibration_mm_s": 1.2, "current_a": 11.0, "temperature_c": 42.5}, ...]
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import numpy as np
import redis.asyncio as redis
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from model_factory import GradCAM, HealthTCN, ResNet34DefectClassifier, load_checkpoint
from websocket_manager import manager

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
RUL_ALERT_THRESHOLD = float(os.getenv("RUL_ALERT_THRESHOLD", "0.30"))
REDIS_STARTUP_RETRIES = int(os.getenv("REDIS_STARTUP_RETRIES", "30"))
REDIS_RETRY_INTERVAL_S = float(os.getenv("REDIS_RETRY_INTERVAL_S", "2"))
TCN_CHECKPOINT = os.getenv("TCN_CHECKPOINT", "")
VISION_CHECKPOINT = os.getenv("VISION_CHECKPOINT", "")
DEVICE_CACHE_PREFIX = "industrial:device:"
logger = logging.getLogger(__name__)


class SensorReading(BaseModel):
    timestamp_s: float | None = None
    vibration_mm_s: float
    current_a: float
    temperature_c: float


class PredictionResponse(BaseModel):
    device_id: str
    timestamp: str
    predicted_rul: float = Field(description="Normalized health/RUL index in [0, 1]")
    predicted_rul_percent: float
    defect_class: str
    defect_probabilities: dict[str, float]
    gradcam_overlay_b64: str = Field(description="Base64-encoded PNG Grad-CAM overlay")
    alert_sent: bool


class AppState:
    redis_client: redis.Redis
    tcn: HealthTCN
    vision: ResNet34DefectClassifier
    device: torch.device


def build_models(device: torch.device) -> tuple[HealthTCN, ResNet34DefectClassifier]:
    """Load local checkpoints when supplied; otherwise create untrained models.

    In production, set TCN_CHECKPOINT and VISION_CHECKPOINT to trained checkpoint
    files created with model_factory.save_checkpoint.
    """
    if TCN_CHECKPOINT:
        tcn, _ = load_checkpoint(TCN_CHECKPOINT, device=device)
        if not isinstance(tcn, HealthTCN):
            raise RuntimeError("TCN_CHECKPOINT does not contain a HealthTCN")
    else:
        tcn = HealthTCN().to(device)

    if VISION_CHECKPOINT:
        vision, _ = load_checkpoint(VISION_CHECKPOINT, device=device)
        if not isinstance(vision, ResNet34DefectClassifier):
            raise RuntimeError("VISION_CHECKPOINT does not contain a ResNet34DefectClassifier")
    else:
        # Avoid implicit network download in an API process. Use a trained checkpoint
        # in production; torchvision is still required to construct the backbone.
        from model_factory import VisionConfig

        vision = ResNet34DefectClassifier(VisionConfig(pretrained=False)).to(device)
    tcn.eval()
    vision.eval()
    return tcn, vision


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcn, vision = await asyncio.to_thread(build_models, device)
    redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
    for attempt in range(1, REDIS_STARTUP_RETRIES + 1):
        try:
            await redis_client.ping()
            break
        except Exception as error:
            if attempt == REDIS_STARTUP_RETRIES:
                await redis_client.aclose()
                raise RuntimeError(
                    f"Unable to connect to Redis at {REDIS_URL} after {REDIS_STARTUP_RETRIES} attempts"
                ) from error
            logger.warning(
                "Redis is not ready (attempt %s/%s); retrying in %s seconds",
                attempt,
                REDIS_STARTUP_RETRIES,
                REDIS_RETRY_INTERVAL_S,
            )
            await asyncio.sleep(REDIS_RETRY_INTERVAL_S)
    else:  # Defensive guard for static analysis; loop either breaks or raises.
        await redis_client.aclose()
        raise RuntimeError(f"Unable to connect to Redis at {REDIS_URL}")
    app.state.services = AppState()
    app.state.services.redis_client = redis_client
    app.state.services.tcn = tcn
    app.state.services.vision = vision
    app.state.services.device = device
    yield
    await redis_client.aclose()


app = FastAPI(title="Industrial Edge Inference API", version="1.0.0", lifespan=lifespan)


@app.get("/health", tags=["operations"])
async def health() -> dict[str, str]:
    """Lightweight readiness endpoint for container platforms and load balancers."""
    return {"status": "ok"}


def parse_sensor_history(raw_json: str, required_steps: int) -> torch.Tensor:
    try:
        raw_items = json.loads(raw_json)
        readings = [SensorReading.model_validate(item) for item in raw_items]
    except (json.JSONDecodeError, ValidationError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid sensor_data JSON: {error}") from error
    if len(readings) < required_steps:
        raise HTTPException(status_code=422, detail=f"sensor_data needs at least {required_steps} readings")
    # Keep the most recent 5 seconds; training and serving must use identical normalization.
    values = [[item.vibration_mm_s, item.current_a, item.temperature_c] for item in readings[-required_steps:]]
    return torch.tensor(values, dtype=torch.float32).unsqueeze(0)


def image_to_tensor(content: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224))
    except Exception as error:
        raise HTTPException(status_code=422, detail="image must be a readable RGB image") from error
    array = np.asarray(image, dtype=np.float32) / 255.0
    # ImageNet normalization required by the ResNet-34 backbone.
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def infer_sync(services: AppState, sensor_history: torch.Tensor, image: torch.Tensor) -> dict[str, Any]:
    """Synchronous CPU/GPU work deliberately executed through asyncio.to_thread."""
    sensor_history = sensor_history.to(services.device)
    image = image.to(services.device)
    with torch.inference_mode():
        rul = float(services.tcn(sensor_history)[0, 0].item())
        logits = services.vision(image)
        probabilities = logits.softmax(dim=1)[0].detach().cpu()
        class_id = int(probabilities.argmax().item())
    # Grad-CAM needs gradients, so it runs outside inference_mode.
    with GradCAM(services.vision) as cam:
        heatmap, _ = cam.generate(image, target_class=class_id)
    # Reverse ImageNet normalization before creating a compact PNG overlay.
    visual = image.detach().cpu()[0]
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    visual = (visual * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    heat = heatmap.detach().cpu()[0].numpy()[..., None]
    colors = np.concatenate((heat, heat**2, 1.0 - heat), axis=2)
    overlay = np.clip((visual * 0.55 + colors * 0.45) * 255, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(overlay, "RGB").save(buffer, format="PNG")
    return {
        "rul": rul,
        "class_id": class_id,
        "probabilities": probabilities.tolist(),
        # Return compact heatmap metadata; persist/serve actual overlay separately if desired.
        "gradcam_peak": float(heatmap.max().item()),
        "gradcam_overlay_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


async def cache_status(client: redis.Redis, device_id: str, status: dict[str, Any]) -> None:
    key = f"{DEVICE_CACHE_PREFIX}{device_id}:status"
    payload = json.dumps(status, ensure_ascii=False)
    async with client.pipeline(transaction=True) as pipe:
        pipe.lpush(key, payload)
        pipe.ltrim(key, 0, 99)
        await pipe.execute()


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    device_id: str = Form(..., min_length=1, max_length=128),
    sensor_data: str = Form(..., description="JSON array of recent PLC readings"),
    image: UploadFile = File(...),
) -> PredictionResponse:
    services: AppState = app.state.services
    sensor_history = parse_sensor_history(
        sensor_data, services.tcn.config.history_seconds * services.tcn.config.sample_rate_hz
    )
    latest_sensor = sensor_history[0, -1].tolist()
    content = await image.read()
    if not content:
        raise HTTPException(status_code=422, detail="Uploaded image is empty")
    image_tensor = image_to_tensor(content)
    result = await asyncio.to_thread(infer_sync, services, sensor_history, image_tensor)

    timestamp = datetime.now(timezone.utc).isoformat()
    class_names = services.vision.config.class_names
    probabilities = {name: round(float(score), 6) for name, score in zip(class_names, result["probabilities"])}
    is_alert = result["rul"] < RUL_ALERT_THRESHOLD
    status = {
        "device_id": device_id,
        "timestamp": timestamp,
        "predicted_rul": round(result["rul"], 6),
        "predicted_rul_percent": round(result["rul"] * 100, 2),
        "defect_class": class_names[result["class_id"]],
        "defect_probabilities": probabilities,
        "vibration_mm_s": round(float(latest_sensor[0]), 4),
        "current_a": round(float(latest_sensor[1]), 4),
        "temperature_c": round(float(latest_sensor[2]), 4),
        "gradcam_peak": round(result["gradcam_peak"], 6),
        "alert_sent": is_alert,
    }
    await cache_status(services.redis_client, device_id, status)
    if is_alert:
        await manager.broadcast_to_device(device_id, {"type": "设备预警", "message": "设备RUL低于阈值", **status})
    return PredictionResponse(**status, gradcam_overlay_b64=result["gradcam_overlay_b64"])


@app.get("/devices/{device_id}/status")
async def latest_device_status(device_id: str, limit: int = 100) -> list[dict[str, Any]]:
    """Return up to the latest 100 cached predictions, newest first."""
    if not 1 <= limit <= 100:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
    services: AppState = app.state.services
    values = await services.redis_client.lrange(f"{DEVICE_CACHE_PREFIX}{device_id}:status", 0, limit - 1)
    return [json.loads(value) for value in values]


@app.websocket("/ws/{device_id}")
async def device_websocket(websocket: WebSocket, device_id: str) -> None:
    await manager.connect(device_id, websocket)
    try:
        await websocket.send_json({"type": "connected", "device_id": device_id})
        while True:
            # Keeps proxies/connections alive; received content is intentionally ignored.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(device_id, websocket)
