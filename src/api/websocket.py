"""WebSocket connection manager for real-time updates."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time dashboard updates."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.chat_connections: dict[str, WebSocket] = {}  # agent_id -> websocket

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def connect_chat(self, agent_id: str, websocket: WebSocket):
        """Connect a chat session for a specific agent."""
        await websocket.accept()
        self.chat_connections[agent_id] = websocket
        logger.info(f"Chat connected for agent: {agent_id}")

    def disconnect_chat(self, agent_id: str):
        """Disconnect a chat session."""
        if agent_id in self.chat_connections:
            del self.chat_connections[agent_id]

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def send_to_agent_chat(self, agent_id: str, message: dict):
        """Send a message to a specific agent's chat connection."""
        if agent_id in self.chat_connections:
            try:
                await self.chat_connections[agent_id].send_text(
                    json.dumps(message, default=str)
                )
            except Exception as e:
                logger.warning(f"Failed to send to chat {agent_id}: {e}")
                self.disconnect_chat(agent_id)

    async def broadcast_agent_update(self, agents: list[dict]):
        """Broadcast agent status updates."""
        await self.broadcast({
            "type": "agents_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": agents,
        })

    async def broadcast_message_event(self, event: dict):
        """Broadcast a message event."""
        await self.broadcast({
            "type": "message_event",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": event,
        })

    async def broadcast_stats(self, stats: dict):
        """Broadcast statistics update."""
        await self.broadcast({
            "type": "stats_update",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": stats,
        })

    async def broadcast_chat_message(self, agent_id: str, role: str, content: str):
        """Broadcast a chat message."""
        await self.broadcast({
            "type": "chat_message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "agent_id": agent_id,
                "role": role,
                "content": content,
            },
        })
