"""Timeline tracking for agent events and activities.

Provides:
- Event recording and storage
- Timeline queries and filtering
- Real-time event streaming
- Activity aggregation and analytics
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of timeline events."""
    # Agent lifecycle
    AGENT_CREATED = "agent_created"
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    AGENT_ERROR = "agent_error"
    AGENT_RESTARTED = "agent_restarted"

    # MCP events
    MCP_CONNECTED = "mcp_connected"
    MCP_DISCONNECTED = "mcp_disconnected"
    MCP_TOOL_CALLED = "mcp_tool_called"
    MCP_TOOL_RESULT = "mcp_tool_result"
    MCP_TOOL_ERROR = "mcp_tool_error"

    # Message events
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_BROADCAST = "message_broadcast"
    MESSAGE_MULTICAST = "message_multicast"

    # Task events
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_DELEGATED = "task_delegated"

    # LLM events
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_STREAM_START = "llm_stream_start"
    LLM_STREAM_END = "llm_stream_end"
    LLM_ERROR = "llm_error"

    # Capability events
    CAPABILITY_INVOKED = "capability_invoked"
    CAPABILITY_COMPLETED = "capability_completed"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONFIG_CHANGED = "config_changed"

    # Industrial/Business Events
    INDUSTRIAL_ACQUISITION = "industrial_acquisition"
    INDUSTRIAL_MERGER = "industrial_merger"
    INDUSTRIAL_IPO = "industrial_ipo"
    INDUSTRIAL_FUNDING = "industrial_funding"
    INDUSTRIAL_BANKRUPTCY = "industrial_bankruptcy"
    INDUSTRIAL_SHUTDOWN = "industrial_shutdown"
    INDUSTRIAL_LAYOFFS = "industrial_layoffs"
    INDUSTRIAL_PARTNERSHIP = "industrial_partnership"
    INDUSTRIAL_PRODUCT_LAUNCH = "industrial_product_launch"
    INDUSTRIAL_REGULATORY = "industrial_regulatory"
    INDUSTRIAL_LEADERSHIP_CHANGE = "industrial_leadership_change"
    INDUSTRIAL_SUPPLY_CHAIN = "industrial_supply_chain"
    INDUSTRIAL_FACTORY = "industrial_factory"
    INDUSTRIAL_CONTRACT = "industrial_contract"

    # GDELT/Geopolitical Events
    GDELT_CONFLICT = "gdelt_conflict"
    GDELT_COOPERATION = "gdelt_cooperation"
    GDELT_DIPLOMATIC = "gdelt_diplomatic"
    GDELT_ECONOMIC = "gdelt_economic"
    GDELT_PROTEST = "gdelt_protest"
    GDELT_MILITARY = "gdelt_military"

    # Emotional Quorum Events
    QUORUM_DELIBERATION_START = "quorum_deliberation_start"
    QUORUM_DELIBERATION_END = "quorum_deliberation_end"
    QUORUM_CONSENSUS_REACHED = "quorum_consensus_reached"
    QUORUM_DISSENT_DETECTED = "quorum_dissent_detected"
    QUORUM_URGENT_SIGNAL = "quorum_urgent_signal"

    # Custom
    CUSTOM = "custom"


class EventSeverity(str, Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class TimelineEvent:
    """A single event in the timeline."""
    event_id: str
    event_type: EventType
    timestamp: datetime
    agent_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.INFO
    title: str = ""
    description: str = ""
    metadata: dict = field(default_factory=dict)
    duration_ms: Optional[float] = None
    parent_event_id: Optional[str] = None
    tags: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "parent_event_id": self.parent_event_id,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEvent":
        return cls(
            event_id=data["event_id"],
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            agent_id=data.get("agent_id"),
            severity=EventSeverity(data.get("severity", "info")),
            title=data.get("title", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
            duration_ms=data.get("duration_ms"),
            parent_event_id=data.get("parent_event_id"),
            tags=set(data.get("tags", [])),
        )


@dataclass
class TimelineQuery:
    """Query parameters for timeline events."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    agent_ids: Optional[List[str]] = None
    event_types: Optional[List[EventType]] = None
    severities: Optional[List[EventSeverity]] = None
    tags: Optional[Set[str]] = None
    limit: int = 100
    offset: int = 0
    order_by: str = "timestamp"
    order_desc: bool = True
    include_metadata: bool = True


@dataclass
class TimelineStats:
    """Statistics about timeline events."""
    total_events: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_agent: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    avg_duration_ms: float = 0.0
    time_range: tuple = field(default_factory=lambda: (None, None))

    def to_dict(self) -> dict:
        return {
            "total_events": self.total_events,
            "events_by_type": self.events_by_type,
            "events_by_agent": self.events_by_agent,
            "events_by_severity": self.events_by_severity,
            "avg_duration_ms": self.avg_duration_ms,
            "time_range": [
                self.time_range[0].isoformat() if self.time_range[0] else None,
                self.time_range[1].isoformat() if self.time_range[1] else None,
            ],
        }


class TimelineStore:
    """In-memory timeline event store with query capabilities."""

    def __init__(self, max_events: int = 100000, retention_hours: int = 24):
        self.max_events = max_events
        self.retention_hours = retention_hours

        # Storage
        self._events: List[TimelineEvent] = []
        self._events_by_id: Dict[str, TimelineEvent] = {}
        self._events_by_agent: Dict[str, List[TimelineEvent]] = defaultdict(list)
        self._events_by_type: Dict[EventType, List[TimelineEvent]] = defaultdict(list)

        # Event listeners
        self._listeners: List[Callable[[TimelineEvent], None]] = []
        self._async_listeners: List[Callable[[TimelineEvent], Any]] = []

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def record(
        self,
        event_type: EventType,
        agent_id: Optional[str] = None,
        title: str = "",
        description: str = "",
        severity: EventSeverity = EventSeverity.INFO,
        metadata: Optional[dict] = None,
        duration_ms: Optional[float] = None,
        parent_event_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> TimelineEvent:
        """Record a new event."""
        event = TimelineEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            agent_id=agent_id,
            severity=severity,
            title=title,
            description=description,
            metadata=metadata or {},
            duration_ms=duration_ms,
            parent_event_id=parent_event_id,
            tags=tags or set(),
        )

        async with self._lock:
            self._events.append(event)
            self._events_by_id[event.event_id] = event

            if agent_id:
                self._events_by_agent[agent_id].append(event)

            self._events_by_type[event_type].append(event)

            # Cleanup if over limit
            if len(self._events) > self.max_events:
                await self._cleanup()

        # Notify listeners
        await self._notify_listeners(event)

        logger.debug(f"Recorded event: {event_type.value} - {title}")
        return event

    async def _cleanup(self):
        """Remove old events."""
        cutoff = datetime.utcnow() - timedelta(hours=self.retention_hours)

        # Remove old events
        self._events = [e for e in self._events if e.timestamp > cutoff]

        # Rebuild indices
        self._events_by_id = {e.event_id: e for e in self._events}
        self._events_by_agent = defaultdict(list)
        self._events_by_type = defaultdict(list)

        for event in self._events:
            if event.agent_id:
                self._events_by_agent[event.agent_id].append(event)
            self._events_by_type[event.event_type].append(event)

    async def query(self, query: TimelineQuery) -> List[TimelineEvent]:
        """Query events with filters."""
        async with self._lock:
            events = self._events.copy()

        # Apply filters
        if query.start_time:
            events = [e for e in events if e.timestamp >= query.start_time]

        if query.end_time:
            events = [e for e in events if e.timestamp <= query.end_time]

        if query.agent_ids:
            events = [e for e in events if e.agent_id in query.agent_ids]

        if query.event_types:
            events = [e for e in events if e.event_type in query.event_types]

        if query.severities:
            events = [e for e in events if e.severity in query.severities]

        if query.tags:
            events = [e for e in events if query.tags & e.tags]

        # Sort
        if query.order_by == "timestamp":
            events.sort(key=lambda e: e.timestamp, reverse=query.order_desc)
        elif query.order_by == "severity":
            severity_order = {s: i for i, s in enumerate(EventSeverity)}
            events.sort(key=lambda e: severity_order[e.severity], reverse=query.order_desc)

        # Pagination
        events = events[query.offset : query.offset + query.limit]

        return events

    async def get_event(self, event_id: str) -> Optional[TimelineEvent]:
        """Get a specific event by ID."""
        return self._events_by_id.get(event_id)

    async def get_agent_timeline(
        self,
        agent_id: str,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[TimelineEvent]:
        """Get timeline for a specific agent."""
        events = self._events_by_agent.get(agent_id, [])

        if since:
            events = [e for e in events if e.timestamp >= since]

        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def get_recent(
        self,
        limit: int = 50,
        event_types: Optional[List[EventType]] = None,
    ) -> List[TimelineEvent]:
        """Get most recent events."""
        events = self._events.copy()

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]

    async def get_stats(
        self,
        since: Optional[datetime] = None,
        agent_id: Optional[str] = None,
    ) -> TimelineStats:
        """Get timeline statistics."""
        events = self._events

        if since:
            events = [e for e in events if e.timestamp >= since]

        if agent_id:
            events = [e for e in events if e.agent_id == agent_id]

        if not events:
            return TimelineStats()

        # Calculate stats
        events_by_type: Dict[str, int] = defaultdict(int)
        events_by_agent: Dict[str, int] = defaultdict(int)
        events_by_severity: Dict[str, int] = defaultdict(int)
        durations = []

        for event in events:
            events_by_type[event.event_type.value] += 1
            if event.agent_id:
                events_by_agent[event.agent_id] += 1
            events_by_severity[event.severity.value] += 1
            if event.duration_ms is not None:
                durations.append(event.duration_ms)

        timestamps = [e.timestamp for e in events]

        return TimelineStats(
            total_events=len(events),
            events_by_type=dict(events_by_type),
            events_by_agent=dict(events_by_agent),
            events_by_severity=dict(events_by_severity),
            avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
            time_range=(min(timestamps), max(timestamps)),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Event Listeners
    # ─────────────────────────────────────────────────────────────────────────

    def add_listener(self, callback: Callable[[TimelineEvent], None]):
        """Add a synchronous event listener."""
        self._listeners.append(callback)

    def add_async_listener(self, callback: Callable[[TimelineEvent], Any]):
        """Add an async event listener."""
        self._async_listeners.append(callback)

    def remove_listener(self, callback: Callable):
        """Remove an event listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
        if callback in self._async_listeners:
            self._async_listeners.remove(callback)

    async def _notify_listeners(self, event: TimelineEvent):
        """Notify all listeners of a new event."""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error in timeline listener: {e}")

        for listener in self._async_listeners:
            try:
                await listener(event)
            except Exception as e:
                logger.error(f"Error in async timeline listener: {e}")


class TimelineTracker:
    """Context manager for tracking operation duration."""

    def __init__(
        self,
        store: TimelineStore,
        event_type: EventType,
        agent_id: Optional[str] = None,
        title: str = "",
        metadata: Optional[dict] = None,
    ):
        self.store = store
        self.event_type = event_type
        self.agent_id = agent_id
        self.title = title
        self.metadata = metadata or {}
        self.start_time: Optional[datetime] = None
        self.event: Optional[TimelineEvent] = None

    async def __aenter__(self) -> "TimelineTracker":
        self.start_time = datetime.utcnow()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000

        severity = EventSeverity.ERROR if exc_type else EventSeverity.INFO
        description = str(exc_val) if exc_val else ""

        self.event = await self.store.record(
            event_type=self.event_type,
            agent_id=self.agent_id,
            title=self.title,
            description=description,
            severity=severity,
            metadata=self.metadata,
            duration_ms=duration_ms,
        )

        return False  # Don't suppress exceptions


# ─────────────────────────────────────────────────────────────────────────────
# Global Timeline Store
# ─────────────────────────────────────────────────────────────────────────────

_timeline_store: Optional[TimelineStore] = None


def get_timeline_store() -> TimelineStore:
    """Get or create the global timeline store."""
    global _timeline_store
    if _timeline_store is None:
        _timeline_store = TimelineStore()
    return _timeline_store


async def record_event(
    event_type: EventType,
    agent_id: Optional[str] = None,
    title: str = "",
    description: str = "",
    severity: EventSeverity = EventSeverity.INFO,
    metadata: Optional[dict] = None,
    **kwargs,
) -> TimelineEvent:
    """Convenience function to record an event to the global store."""
    store = get_timeline_store()
    return await store.record(
        event_type=event_type,
        agent_id=agent_id,
        title=title,
        description=description,
        severity=severity,
        metadata=metadata,
        **kwargs,
    )


def track(
    event_type: EventType,
    agent_id: Optional[str] = None,
    title: str = "",
    metadata: Optional[dict] = None,
) -> TimelineTracker:
    """Create a timeline tracker context manager."""
    store = get_timeline_store()
    return TimelineTracker(
        store=store,
        event_type=event_type,
        agent_id=agent_id,
        title=title,
        metadata=metadata,
    )
