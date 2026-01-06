"""Event handlers for different event types."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from .types import (
    Event,
    EventType,
    WorldEvent,
    WorldEventCategory,
    AgentEvent,
    AgentEventCategory,
    Message,
    MessageType,
)

logger = logging.getLogger(__name__)


class EventHandler(ABC):
    """Base class for event handlers."""

    def __init__(self, handler_id: str):
        self.handler_id = handler_id
        self.handled_count = 0
        self.last_handled: Optional[datetime] = None

    @abstractmethod
    async def handle(self, event: Event) -> Any:
        """Handle an event."""
        pass

    async def __call__(self, event: Event) -> Any:
        """Make handler callable."""
        self.handled_count += 1
        self.last_handled = datetime.now()
        return await self.handle(event)


class WorldEventHandler(EventHandler):
    """Handler for world events."""

    def __init__(
        self,
        handler_id: str,
        categories: Optional[set[WorldEventCategory]] = None,
        min_significance: float = 0.0,
    ):
        super().__init__(handler_id)
        self.categories = categories or set()
        self.min_significance = min_significance

        # Callbacks for different processing stages
        self._on_receive: Optional[Callable] = None
        self._on_analyze: Optional[Callable] = None
        self._on_integrate: Optional[Callable] = None

    def on_receive(self, fn: Callable) -> Callable:
        """Decorator for receive callback."""
        self._on_receive = fn
        return fn

    def on_analyze(self, fn: Callable) -> Callable:
        """Decorator for analyze callback."""
        self._on_analyze = fn
        return fn

    def on_integrate(self, fn: Callable) -> Callable:
        """Decorator for integrate callback."""
        self._on_integrate = fn
        return fn

    async def handle(self, event: Event) -> dict:
        """Handle a world event through receive -> analyze -> integrate pipeline."""
        if not isinstance(event, WorldEvent):
            return {"skipped": True, "reason": "not a world event"}

        # Filter by category
        if self.categories and event.category not in self.categories:
            return {"skipped": True, "reason": "category filtered"}

        # Filter by significance
        if event.significance < self.min_significance:
            return {"skipped": True, "reason": "below significance threshold"}

        result = {"event_id": event.event_id, "stages": {}}

        # Stage 1: Receive
        if self._on_receive:
            try:
                r = self._on_receive(event)
                if asyncio.iscoroutine(r):
                    r = await r
                result["stages"]["receive"] = r
            except Exception as e:
                logger.error(f"Error in receive stage: {e}")
                result["stages"]["receive"] = {"error": str(e)}

        # Stage 2: Analyze
        if self._on_analyze:
            try:
                r = self._on_analyze(event, result.get("stages", {}).get("receive"))
                if asyncio.iscoroutine(r):
                    r = await r
                result["stages"]["analyze"] = r
            except Exception as e:
                logger.error(f"Error in analyze stage: {e}")
                result["stages"]["analyze"] = {"error": str(e)}

        # Stage 3: Integrate
        if self._on_integrate:
            try:
                r = self._on_integrate(event, result.get("stages", {}))
                if asyncio.iscoroutine(r):
                    r = await r
                result["stages"]["integrate"] = r
            except Exception as e:
                logger.error(f"Error in integrate stage: {e}")
                result["stages"]["integrate"] = {"error": str(e)}

        return result


class AgentEventHandler(EventHandler):
    """Handler for internal agent events."""

    def __init__(
        self,
        handler_id: str,
        categories: Optional[set[AgentEventCategory]] = None,
        agent_filter: Optional[set[str]] = None,  # Filter by agent IDs
    ):
        super().__init__(handler_id)
        self.categories = categories or set()
        self.agent_filter = agent_filter

        # Track agent states
        self._agent_states: dict[str, dict] = {}

        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_reasoning: Optional[Callable] = None
        self._on_decision: Optional[Callable] = None

    def on_state_change(self, fn: Callable) -> Callable:
        """Decorator for state change callback."""
        self._on_state_change = fn
        return fn

    def on_reasoning(self, fn: Callable) -> Callable:
        """Decorator for reasoning event callback."""
        self._on_reasoning = fn
        return fn

    def on_decision(self, fn: Callable) -> Callable:
        """Decorator for decision callback."""
        self._on_decision = fn
        return fn

    async def handle(self, event: Event) -> dict:
        """Handle an agent event."""
        if not isinstance(event, AgentEvent):
            return {"skipped": True, "reason": "not an agent event"}

        # Filter by agent
        if self.agent_filter and event.agent_id not in self.agent_filter:
            return {"skipped": True, "reason": "agent filtered"}

        # Filter by category
        if self.categories and event.category not in self.categories:
            return {"skipped": True, "reason": "category filtered"}

        result = {"event_id": event.event_id, "agent_id": event.agent_id}

        # Route to appropriate callback
        if event.category in {
            AgentEventCategory.STATE_CHANGE,
            AgentEventCategory.BELIEF_UPDATE,
            AgentEventCategory.GOAL_UPDATE,
        }:
            if self._on_state_change:
                try:
                    r = self._on_state_change(event)
                    if asyncio.iscoroutine(r):
                        r = await r
                    result["state_change"] = r
                except Exception as e:
                    logger.error(f"Error in state change handler: {e}")
                    result["error"] = str(e)

            # Track state
            if event.new_state:
                self._agent_states[event.agent_id] = event.new_state

        elif event.category in {
            AgentEventCategory.REASONING_START,
            AgentEventCategory.REASONING_STEP,
            AgentEventCategory.REASONING_COMPLETE,
        }:
            if self._on_reasoning:
                try:
                    r = self._on_reasoning(event)
                    if asyncio.iscoroutine(r):
                        r = await r
                    result["reasoning"] = r
                except Exception as e:
                    logger.error(f"Error in reasoning handler: {e}")
                    result["error"] = str(e)

        elif event.category in {
            AgentEventCategory.DECISION_MADE,
            AgentEventCategory.ACTION_TAKEN,
        }:
            if self._on_decision:
                try:
                    r = self._on_decision(event)
                    if asyncio.iscoroutine(r):
                        r = await r
                    result["decision"] = r
                except Exception as e:
                    logger.error(f"Error in decision handler: {e}")
                    result["error"] = str(e)

        return result

    def get_agent_state(self, agent_id: str) -> Optional[dict]:
        """Get tracked state for an agent."""
        return self._agent_states.get(agent_id)


class MessageHandler(EventHandler):
    """Handler for agent-to-agent messages."""

    def __init__(
        self,
        handler_id: str,
        agent_id: str,  # This handler's agent ID
        message_types: Optional[set[MessageType]] = None,
    ):
        super().__init__(handler_id)
        self.agent_id = agent_id
        self.message_types = message_types or set()

        # Inbox
        self._inbox: list[Message] = []
        self._max_inbox = 1000

        # Callbacks
        self._on_message: Optional[Callable] = None
        self._on_query: Optional[Callable] = None
        self._on_request: Optional[Callable] = None

    def on_message(self, fn: Callable) -> Callable:
        """Decorator for general message callback."""
        self._on_message = fn
        return fn

    def on_query(self, fn: Callable) -> Callable:
        """Decorator for query callback (should return response)."""
        self._on_query = fn
        return fn

    def on_request(self, fn: Callable) -> Callable:
        """Decorator for request callback (should return accept/reject)."""
        self._on_request = fn
        return fn

    async def handle(self, event: Event) -> dict:
        """Handle a message."""
        if not isinstance(event, Message):
            return {"skipped": True, "reason": "not a message"}

        # Check if message is for us (or broadcast)
        is_broadcast = event.message_type == MessageType.BROADCAST
        is_for_us = event.recipient_id == self.agent_id

        if not is_broadcast and not is_for_us:
            return {"skipped": True, "reason": "not addressed to us"}

        # Filter by message type
        if self.message_types and event.message_type not in self.message_types:
            return {"skipped": True, "reason": "message type filtered"}

        # Add to inbox
        self._inbox.append(event)
        if len(self._inbox) > self._max_inbox:
            self._inbox = self._inbox[-self._max_inbox:]

        result = {
            "event_id": event.event_id,
            "from": event.sender_name or event.sender_id,
            "type": event.message_type.value,
        }

        # Route to appropriate callback
        if event.message_type == MessageType.QUERY and self._on_query:
            try:
                r = self._on_query(event)
                if asyncio.iscoroutine(r):
                    r = await r
                result["response"] = r
            except Exception as e:
                logger.error(f"Error in query handler: {e}")
                result["error"] = str(e)

        elif event.message_type == MessageType.REQUEST and self._on_request:
            try:
                r = self._on_request(event)
                if asyncio.iscoroutine(r):
                    r = await r
                result["decision"] = r
            except Exception as e:
                logger.error(f"Error in request handler: {e}")
                result["error"] = str(e)

        elif self._on_message:
            try:
                r = self._on_message(event)
                if asyncio.iscoroutine(r):
                    r = await r
                result["handled"] = r
            except Exception as e:
                logger.error(f"Error in message handler: {e}")
                result["error"] = str(e)

        return result

    def get_inbox(
        self,
        limit: int = 50,
        message_type: Optional[MessageType] = None,
    ) -> list[Message]:
        """Get messages from inbox."""
        msgs = self._inbox

        if message_type:
            msgs = [m for m in msgs if m.message_type == message_type]

        return msgs[-limit:]

    def get_conversation(self, conversation_id: str) -> list[Message]:
        """Get all messages in a conversation."""
        return [
            m for m in self._inbox
            if m.conversation_id == conversation_id or m.event_id == conversation_id
        ]
