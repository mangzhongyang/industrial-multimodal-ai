"""Minimal Streamlit industrial dashboard.

Install and run:
    pip install streamlit streamlit-autorefresh requests websocket-client pandas altair
    streamlit run dashboard.py

Set API_URL to point at the FastAPI server, e.g.:
    set API_URL=http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_autorefresh import st_autorefresh

try:
    import websocket
except ImportError:
    websocket = None


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
if not API_URL.startswith(("http://", "https://")):
    API_URL = f"http://{API_URL}"
REFRESH_MS = int(os.getenv("DASHBOARD_REFRESH_MS", "3000"))
DEMO_IMAGE_PATH = Path(__file__).parent / "predict_sample" / "images" / "sample_000019.png"


def api_get(path: str) -> Any | None:
    try:
        response = requests.get(f"{API_URL}{path}", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        st.warning(f"无法连接后端：{error}")
        return None


@st.cache_resource(show_spinner=False)
def get_websocket_collector(device_id: str) -> queue.Queue[dict[str, Any]] | None:
    """Maintain one process-wide WebSocket per device across Streamlit reruns."""
    if websocket is None:
        return None
    message_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    ws_url = API_URL.replace("http://", "ws://").replace("https://", "wss://") + f"/ws/{device_id}"

    def listen() -> None:
        while True:
            connection = None
            try:
                connection = websocket.create_connection(ws_url, timeout=10)
                while True:
                    payload = json.loads(connection.recv())
                    if payload.get("type") == "璁惧棰勮":
                        message_queue.put(payload)
            except Exception:
                time.sleep(3)
            finally:
                if connection is not None:
                    connection.close()

    threading.Thread(target=listen, daemon=True, name=f"alert-ws-{device_id}").start()
    return message_queue


def start_websocket_collector(device_id: str) -> None:
    """Start one background subscriber per selected device for this browser session."""
    message_queue = get_websocket_collector(device_id)
    if message_queue is None:
        return
    if st.session_state.get("ws_device") != device_id:
        st.session_state.ws_device = device_id
        st.session_state.alerts = []
    st.session_state.alert_queue = message_queue


def drain_alerts() -> None:
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    alerts: list[dict[str, Any]] = st.session_state.alerts
    messages: queue.Queue[dict[str, Any]] | None = st.session_state.get("alert_queue")
    if messages is None:
        return
    while True:
        try:
            alerts.insert(0, messages.get_nowait())
        except queue.Empty:
            break
    del alerts[100:]


def overview_tab(device_id: str) -> None:
    st.subheader("产线总览")
    statuses = api_get(f"/devices/{device_id}/status?limit=100")
    if not statuses:
        st.info("暂无状态数据。请先通过 /predict 提交生产样本。")
        return
    frame = pd.DataFrame(statuses)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp")
    metrics = [field for field in ("temperature_c", "vibration_mm_s") if field in frame.columns]
    if not metrics:
        st.info("当前后端缓存仅保存预测结果；提交时保存传感器快照后即可显示温度/振动趋势。")
        st.dataframe(frame[["timestamp", "predicted_rul_percent", "defect_class"]], use_container_width=True)
        return
    long = frame.melt(
        id_vars=["timestamp", "predicted_rul_percent"], value_vars=metrics, var_name="metric", value_name="value"
    )
    lines = alt.Chart(long).mark_line().encode(x="timestamp:T", y="value:Q", color="metric:N")
    anomalies = frame[frame["predicted_rul_percent"] < 30].melt(
        id_vars=["timestamp"], value_vars=metrics, var_name="metric", value_name="value"
    )
    points = alt.Chart(anomalies).mark_point(color="red", size=90).encode(x="timestamp:T", y="value:Q")
    st.altair_chart((lines + points).properties(height=360), use_container_width=True)
    st.caption("红点：RUL 低于 30% 的异常预测。")


def cloud_demo_card() -> None:
    """Show a bundled example so portfolio visitors see results immediately."""
    st.markdown("#### 云端预置演示案例")
    st.caption("无需上传或调用接口：此为固定合成样品及其已验证的演示推理结果。")
    if not DEMO_IMAGE_PATH.exists():
        st.info("演示图片将在下一次部署中可用。")
        return

    original = Image.open(DEMO_IMAGE_PATH).convert("RGB")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)
    # Coordinates are the YOLO labels for this reproducible, seed-fixed sample.
    defects = [
        ("划痕", (140, 260, 164, 347), "#e74c3c"),
        ("脏污", (359, 37, 441, 131), "#f1c40f"),
    ]
    for name, box, color in defects:
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0], max(0, box[1] - 20)), name, fill=color)

    left, right = st.columns(2)
    left.image(original, caption="合成产品表面样品：划痕 + 脏污", use_container_width=True)
    right.image(annotated, caption="YOLO 缺陷标注可视化", use_container_width=True)
    rul, defect, alert = st.columns(3)
    rul.metric("预测 RUL", "0.3%")
    defect.metric("视觉分类", "脏污（stain）")
    alert.error("设备预警：RUL 低于 30%")
    st.caption("演示结果：good 0.0 · scratch 0.0 · stain 1.0")


def visual_tab(device_id: str) -> None:
    st.subheader("视觉质检")
    cloud_demo_card()
    st.divider()
    st.markdown("#### 在线提交新样品")
    upload = st.file_uploader("上传产品表面图片", type=["png", "jpg", "jpeg"])
    default_history = json.dumps(
        [
            {"timestamp_s": i, "vibration_mm_s": 1.2 + i * 0.05, "current_a": 11.0, "temperature_c": 42.0 + i * 0.1}
            for i in range(5)
        ],
        ensure_ascii=False,
        indent=2,
    )
    sensor_json = st.text_area("过去 5 秒传感器 JSON", value=default_history, height=160)
    if upload and st.button("开始质检", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/predict",
                data={"device_id": device_id, "sensor_data": sensor_json},
                files={"image": (upload.name, upload.getvalue(), upload.type or "image/png")},
                timeout=45,
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as error:
            st.error(f"推理失败：{error}")
            return
        left, right = st.columns(2)
        left.image(upload, caption="原始产品图")
        right.image(base64.b64decode(result["gradcam_overlay_b64"]), caption="Grad-CAM 热力图叠加")
        st.success(f"分类：{result['defect_class']}｜RUL：{result['predicted_rul_percent']}%")
        st.json(result["defect_probabilities"])


def log_tab() -> None:
    st.subheader("日志中心")
    drain_alerts()
    alerts = st.session_state.get("alerts", [])
    if alerts:
        table = pd.DataFrame(alerts)
        columns = [
            name
            for name in ("timestamp", "device_id", "message", "predicted_rul_percent", "defect_class")
            if name in table
        ]
        st.dataframe(table[columns], use_container_width=True, hide_index=True)
    else:
        st.info("尚未收到 WebSocket 设备预警。")


st.set_page_config(page_title="工业 AI 仪表盘", page_icon="🏭", layout="wide")
st.title("🏭 智能制造 AI 边缘仪表盘")
device_id = st.sidebar.text_input("设备 ID", value="PLC-001")
st.sidebar.markdown(f"API 文档：[打开 /docs]({API_URL}/docs)")
st_autorefresh(interval=REFRESH_MS, key="industrial-dashboard-refresh")
start_websocket_collector(device_id)

tab_overview, tab_visual, tab_logs = st.tabs(["产线总览", "视觉质检", "日志中心"])
with tab_overview:
    overview_tab(device_id)
with tab_visual:
    visual_tab(device_id)
with tab_logs:
    log_tab()
