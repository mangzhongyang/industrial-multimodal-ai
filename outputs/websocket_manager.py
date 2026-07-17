"""Connection registry for device-scoped FastAPI WebSocket notifications."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi import WebSocket


class ConnectionManager:
    """Stores sockets by device ID and safely broadcasts JSON messages."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, device_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections[device_id].add(websocket)

    async def disconnect(self, device_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            connections = self._connections.get(device_id)
            if not connections:
                return
            connections.discard(websocket)
            if not connections:
                self._connections.pop(device_id, None)

    async def broadcast_to_device(self, device_id: str, message: dict[str, object]) -> int:
        """Broadcast a JSON-compatible message and drop stale connections."""
        async with self._lock:
            recipients = list(self._connections.get(device_id, set()))
        results = await asyncio.gather(*(socket.send_json(message) for socket in recipients), return_exceptions=True)
        for socket, result in zip(recipients, results):
            if isinstance(result, Exception):
                await self.disconnect(device_id, socket)
        return sum(not isinstance(result, Exception) for result in results)


manager = ConnectionManager()
