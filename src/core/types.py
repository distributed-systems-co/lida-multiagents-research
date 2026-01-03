"""Core type definitions for the multi-agent system."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Optional, TypeVar, Generic
import uuid


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(str, Enum):
    """All supported message types in the system."""

    # ─── Point-to-Point ───
    DIRECT = "direct"

    # ─── One-to-Many ───
    BROADCAST = "broadcast"      # To ALL agents
    MULTICAST = "multicast"      # To topic subscribers
    ANYCAST = "anycast"          # To ONE of many (load balanced)

    # ─── Request/Response ───
    REQUEST = "request"
    RESPONSE = "response"

    # ─── Coordination ───
    DELEGATE = "delegate"        # Assign work
    REPORT = "report"            # Report results

    # ─── Consensus ───
    PROPOSE = "propose"
    VOTE = "vote"
    COMMIT = "commit"
    ABORT = "abort"

    # ─── Scatter/Gather ───
    SCATTER = "scatter"          # Fan-out request
    GATHER = "gather"            # Aggregated response

    # ─── Lifecycle ───
    SPAWN = "spawn"
    TERMINATE = "terminate"
    SUSPEND = "suspend"
    RESUME = "resume"
    HEARTBEAT = "heartbeat"

    # ─── System ───
    ACK = "ack"
    NACK = "nack"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"

    # ─── Events ───
    EVENT = "event"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"


class Priority(int, Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    SYSTEM = 4  # Reserved for system messages


class DeliveryGuarantee(str, Enum):
    """Message delivery guarantees."""
    AT_MOST_ONCE = "at_most_once"    # Fire and forget
    AT_LEAST_ONCE = "at_least_once"  # With retries
    EXACTLY_ONCE = "exactly_once"    # With deduplication


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AgentStatus(str, Enum):
    """Agent lifecycle states."""
    INITIALIZING = "initializing"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    BUSY = "busy"
    SUSPENDED = "suspended"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"
    DEAD = "dead"


class AgentRole(str, Enum):
    """Predefined agent roles."""
    ORCHESTRATOR = "orchestrator"   # Coordinates other agents
    WORKER = "worker"               # Performs tasks
    MONITOR = "monitor"             # Observes and reports
    ROUTER = "router"               # Routes messages
    GATEWAY = "gateway"             # External interface
    SUPERVISOR = "supervisor"       # Manages agent lifecycle
    PERSONA = "persona"             # Domain expert


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

class RoutingStrategy(str, Enum):
    """Message routing strategies."""
    DIRECT = "direct"               # Specific recipient
    ROUND_ROBIN = "round_robin"     # Rotate through targets
    RANDOM = "random"               # Random selection
    LEAST_LOADED = "least_loaded"   # Lowest queue depth
    CONSISTENT_HASH = "hash"        # Content-based routing
    BROADCAST = "broadcast"         # All targets
    FIRST_AVAILABLE = "first"       # First responding


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Address:
    """Agent address for routing."""
    agent_id: str
    node_id: Optional[str] = None   # For distributed deployment
    channel: Optional[str] = None   # Specific channel

    def __str__(self) -> str:
        parts = [self.agent_id]
        if self.node_id:
            parts.insert(0, self.node_id)
        if self.channel:
            parts.append(self.channel)
        return "/".join(parts)

    @classmethod
    def parse(cls, addr: str) -> "Address":
        parts = addr.split("/")
        if len(parts) == 1:
            return cls(agent_id=parts[0])
        elif len(parts) == 2:
            return cls(node_id=parts[0], agent_id=parts[1])
        else:
            return cls(node_id=parts[0], agent_id=parts[1], channel="/".join(parts[2:]))


@dataclass
class Topic:
    """Pub/sub topic definition."""
    name: str
    pattern: Optional[str] = None   # Glob pattern for matching
    retention: int = 1000           # Messages to retain

    def matches(self, other: str) -> bool:
        if self.pattern:
            import fnmatch
            return fnmatch.fnmatch(other, self.pattern)
        return self.name == other


@dataclass
class Subscription:
    """Topic subscription."""
    topic: str
    subscriber_id: str
    filter_fn: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MessageStats:
    """Message statistics."""
    sent: int = 0
    received: int = 0
    processed: int = 0
    failed: int = 0
    dropped: int = 0
    retried: int = 0
    avg_latency_ms: float = 0.0

    def __add__(self, other: "MessageStats") -> "MessageStats":
        return MessageStats(
            sent=self.sent + other.sent,
            received=self.received + other.received,
            processed=self.processed + other.processed,
            failed=self.failed + other.failed,
            dropped=self.dropped + other.dropped,
            retried=self.retried + other.retried,
        )


@dataclass
class AgentStats:
    """Agent statistics."""
    agent_id: str
    status: AgentStatus
    uptime_seconds: float = 0.0
    messages: MessageStats = field(default_factory=MessageStats)
    inbox_depth: int = 0
    outbox_depth: int = 0
    last_activity: Optional[datetime] = None
    error_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

T = TypeVar('T')

@dataclass
class Result(Generic[T]):
    """Result type for operations that can fail."""
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        return cls(success=True, value=value)

    @classmethod
    def fail(cls, error: str) -> "Result[T]":
        return cls(success=False, error=error)

    def unwrap(self) -> T:
        if not self.success:
            raise ValueError(f"Result failed: {self.error}")
        return self.value

    def unwrap_or(self, default: T) -> T:
        return self.value if self.success else default
