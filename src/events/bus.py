"""Event bus for publishing and subscribing to events."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, Union
from weakref import WeakSet

from .types import (
    Event,
    EventType,
    EventPriority,
    WorldEvent,
    WorldEventCategory,
    AgentEvent,
    AgentEventCategory,
    Message,
    MessageType,
)

logger = logging.getLogger(__name__)

# Type for event handlers
EventCallback = Callable[[Event], Any]
AsyncEventCallback = Callable[[Event], Any]  # Can be sync or async


@dataclass
class EventSubscription:
    """A subscription to events."""

    subscription_id: str
    subscriber_id: str
    event_types: set[EventType] = field(default_factory=set)
    categories: set[str] = field(default_factory=set)  # WorldEventCategory, AgentEventCategory, or MessageType
    callback: Optional[AsyncEventCallback] = None
    filter_fn: Optional[Callable[[Event], bool]] = None
    priority: EventPriority = EventPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    active: bool = True

    def matches(self, event: Event) -> bool:
        """Check if this subscription matches an event."""
        if not self.active:
            return False

        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        # Check category
        if self.categories:
            category = None
            if isinstance(event, WorldEvent):
                category = event.category.value
            elif isinstance(event, AgentEvent):
                category = event.category.value
            elif isinstance(event, Message):
                category = event.message_type.value

            if category and category not in self.categories:
                return False

        # Apply custom filter
        if self.filter_fn and not self.filter_fn(event):
            return False

        return True


class EventBus:
    """Central event bus for the multi-agent system."""

    def __init__(self, max_history: int = 1000):
        self._subscriptions: dict[str, EventSubscription] = {}
        self._subscriber_subs: dict[str, set[str]] = defaultdict(set)

        # Event queues by priority
        self._queues: dict[EventPriority, asyncio.Queue] = {
            p: asyncio.Queue() for p in EventPriority
        }

        # Event history
        self._history: list[Event] = []
        self._max_history = max_history

        # Processing state
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_dropped": 0,
        }

    async def start(self):
        """Start the event bus processor."""
        if self._running:
            return

        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        logger.info("EventBus started")

    async def stop(self):
        """Stop the event bus processor."""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("EventBus stopped")

    async def _process_events(self):
        """Process events from queues."""
        while self._running:
            # Process high priority first
            for priority in reversed(list(EventPriority)):
                queue = self._queues[priority]
                while not queue.empty():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.01)
                        await self._deliver_event(event)
                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        logger.error(f"Error processing event: {e}")

            await asyncio.sleep(0.01)  # Prevent busy loop

    async def _deliver_event(self, event: Event):
        """Deliver an event to all matching subscribers."""
        delivered = 0

        for sub in self._subscriptions.values():
            if sub.matches(event):
                try:
                    if sub.callback:
                        result = sub.callback(event)
                        if asyncio.iscoroutine(result):
                            await result
                        delivered += 1
                except Exception as e:
                    logger.error(f"Error in subscriber {sub.subscriber_id}: {e}")

        self._stats["events_delivered"] += delivered

        if delivered == 0:
            self._stats["events_dropped"] += 1

    def subscribe(
        self,
        subscriber_id: str,
        callback: AsyncEventCallback,
        event_types: Optional[set[EventType]] = None,
        categories: Optional[set[str]] = None,
        filter_fn: Optional[Callable[[Event], bool]] = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> str:
        """Subscribe to events.

        Args:
            subscriber_id: ID of the subscribing agent/component
            callback: Function to call when event matches
            event_types: Set of EventType to subscribe to (None = all)
            categories: Set of category values to filter by
            filter_fn: Additional filter function
            priority: Subscription priority for delivery order

        Returns:
            subscription_id
        """
        import uuid

        sub_id = str(uuid.uuid4())[:8]

        subscription = EventSubscription(
            subscription_id=sub_id,
            subscriber_id=subscriber_id,
            event_types=event_types or set(),
            categories=categories or set(),
            callback=callback,
            filter_fn=filter_fn,
            priority=priority,
        )

        self._subscriptions[sub_id] = subscription
        self._subscriber_subs[subscriber_id].add(sub_id)

        logger.debug(f"Subscription {sub_id} created for {subscriber_id}")
        return sub_id

    def unsubscribe(self, subscription_id: str):
        """Remove a subscription."""
        if subscription_id in self._subscriptions:
            sub = self._subscriptions.pop(subscription_id)
            self._subscriber_subs[sub.subscriber_id].discard(subscription_id)
            logger.debug(f"Subscription {subscription_id} removed")

    def unsubscribe_all(self, subscriber_id: str):
        """Remove all subscriptions for a subscriber."""
        for sub_id in list(self._subscriber_subs[subscriber_id]):
            self.unsubscribe(sub_id)

    async def publish(self, event: Event):
        """Publish an event to the bus."""
        # Add to history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Queue for processing
        queue = self._queues[event.priority]
        await queue.put(event)

        self._stats["events_published"] += 1
        logger.debug(f"Event {event.event_id[:8]} published: {event.event_type.value}")

    def publish_sync(self, event: Event):
        """Synchronously publish an event (queues for async processing)."""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        queue = self._queues[event.priority]
        try:
            queue.put_nowait(event)
            self._stats["events_published"] += 1
        except asyncio.QueueFull:
            logger.warning(f"Event queue full, dropping event {event.event_id}")
            self._stats["events_dropped"] += 1

    async def publish_world_event(
        self,
        title: str,
        description: str,
        category: WorldEventCategory = WorldEventCategory.OBSERVATION,
        content: Any = None,
        significance: float = 0.5,
        source: str = "system",
        **kwargs,
    ) -> WorldEvent:
        """Convenience method to publish a world event."""
        event = WorldEvent(
            category=category,
            title=title,
            description=description,
            content=content,
            significance=significance,
            source=source,
            **kwargs,
        )
        await self.publish(event)
        return event

    async def publish_agent_event(
        self,
        agent_id: str,
        agent_name: str,
        category: AgentEventCategory,
        action: str,
        details: Any = None,
        **kwargs,
    ) -> AgentEvent:
        """Convenience method to publish an agent event."""
        event = AgentEvent(
            category=category,
            agent_id=agent_id,
            agent_name=agent_name,
            action=action,
            details=details,
            source=agent_id,
            **kwargs,
        )
        await self.publish(event)
        return event

    async def send_message(
        self,
        sender_id: str,
        sender_name: str,
        recipient_id: str,
        recipient_name: str,
        content: Any,
        message_type: MessageType = MessageType.INFORM,
        subject: str = "",
        **kwargs,
    ) -> Message:
        """Send a message between agents."""
        msg = Message(
            message_type=message_type,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            subject=subject,
            content=content,
            source=sender_id,
            **kwargs,
        )
        await self.publish(msg)
        return msg

    async def broadcast(
        self,
        sender_id: str,
        sender_name: str,
        content: Any,
        subject: str = "",
        **kwargs,
    ) -> Message:
        """Broadcast a message to all agents."""
        msg = Message(
            message_type=MessageType.BROADCAST,
            sender_id=sender_id,
            sender_name=sender_name,
            recipient_id="",  # Empty = broadcast
            recipient_name="",
            subject=subject,
            content=content,
            source=sender_id,
            **kwargs,
        )
        await self.publish(msg)
        return msg

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> list[Event]:
        """Get event history."""
        events = self._history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events[-limit:]

    def get_stats(self) -> dict:
        """Get bus statistics."""
        return {
            **self._stats,
            "subscriptions": len(self._subscriptions),
            "history_size": len(self._history),
            "queue_sizes": {p.name: self._queues[p].qsize() for p in EventPriority},
        }


# Global event bus instance
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
