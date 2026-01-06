"""Event type definitions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Optional


class EventType(str, Enum):
    """Top-level event categories."""
    WORLD = "world"
    AGENT = "agent"
    MESSAGE = "message"


class EventPriority(int, Enum):
    """Event processing priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class WorldEventCategory(str, Enum):
    """Categories of world events."""
    # Information events
    NEWS = "news"
    DISCOVERY = "discovery"
    ANNOUNCEMENT = "announcement"

    # State changes
    MARKET_CHANGE = "market_change"
    ENVIRONMENT_CHANGE = "environment_change"
    POLICY_CHANGE = "policy_change"

    # Social events
    CONFLICT = "conflict"
    COOPERATION = "cooperation"
    ELECTION = "election"

    # Technical events
    SYSTEM_UPDATE = "system_update"
    DATA_RELEASE = "data_release"

    # Generic
    OBSERVATION = "observation"
    CUSTOM = "custom"


class AgentEventCategory(str, Enum):
    """Categories of internal agent events."""
    # Cognitive events
    REASONING_START = "reasoning_start"
    REASONING_STEP = "reasoning_step"
    REASONING_COMPLETE = "reasoning_complete"

    # State changes
    STATE_CHANGE = "state_change"
    BELIEF_UPDATE = "belief_update"
    GOAL_UPDATE = "goal_update"

    # Decisions
    DECISION_MADE = "decision_made"
    ACTION_TAKEN = "action_taken"

    # Learning
    LEARNING_EVENT = "learning_event"
    HYPOTHESIS_FORMED = "hypothesis_formed"
    HYPOTHESIS_VERIFIED = "hypothesis_verified"

    # Errors
    ERROR = "error"
    RECOVERY = "recovery"


class MessageType(str, Enum):
    """Types of agent-to-agent messages."""
    # Communication
    QUERY = "query"
    RESPONSE = "response"
    INFORM = "inform"

    # Coordination
    REQUEST = "request"
    PROPOSE = "propose"
    ACCEPT = "accept"
    REJECT = "reject"

    # Collaboration
    DELEGATE = "delegate"
    REPORT = "report"

    # Discussion
    ARGUE = "argue"
    AGREE = "agree"
    DISAGREE = "disagree"

    # Signals
    PING = "ping"
    ACK = "ack"
    BROADCAST = "broadcast"


@dataclass
class Event:
    """Base event class."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.WORLD
    timestamp: datetime = field(default_factory=datetime.now)
    priority: EventPriority = EventPriority.NORMAL
    source: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "source": self.source,
            "metadata": self.metadata,
        }


@dataclass
class WorldEvent(Event):
    """External world event that agents observe."""

    event_type: EventType = field(default=EventType.WORLD, init=False)
    category: WorldEventCategory = WorldEventCategory.OBSERVATION

    # Event content
    title: str = ""
    description: str = ""
    content: Any = None

    # Context
    location: Optional[str] = None
    domain: Optional[str] = None
    entities: list[str] = field(default_factory=list)

    # Impact
    significance: float = 0.5  # 0-1 scale
    confidence: float = 1.0    # How certain is this event real

    # Temporal
    event_time: Optional[datetime] = None  # When the event actually occurred
    duration: Optional[float] = None       # Duration in seconds if applicable

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "category": self.category.value,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "location": self.location,
            "domain": self.domain,
            "entities": self.entities,
            "significance": self.significance,
            "confidence": self.confidence,
            "event_time": self.event_time.isoformat() if self.event_time else None,
            "duration": self.duration,
        })
        return base


@dataclass
class AgentEvent(Event):
    """Internal agent event."""

    event_type: EventType = field(default=EventType.AGENT, init=False)
    category: AgentEventCategory = AgentEventCategory.STATE_CHANGE

    # Agent info
    agent_id: str = ""
    agent_name: str = ""

    # Event details
    action: str = ""
    details: Any = None

    # State tracking
    previous_state: Optional[dict] = None
    new_state: Optional[dict] = None

    # Reasoning context
    trace_id: Optional[str] = None
    step_id: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "category": self.category.value,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action": self.action,
            "details": self.details,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "trace_id": self.trace_id,
            "step_id": self.step_id,
            "confidence": self.confidence,
        })
        return base


@dataclass
class Message(Event):
    """Agent-to-agent message."""

    event_type: EventType = field(default=EventType.MESSAGE, init=False)
    message_type: MessageType = MessageType.INFORM

    # Routing
    sender_id: str = ""
    sender_name: str = ""
    recipient_id: str = ""           # Empty for broadcast
    recipient_name: str = ""

    # Content
    subject: str = ""
    content: Any = None

    # Conversation tracking
    conversation_id: Optional[str] = None
    reply_to: Optional[str] = None   # message_id being replied to

    # Delivery
    requires_ack: bool = False
    ttl: Optional[float] = None      # Time to live in seconds

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "sender_name": self.sender_name,
            "recipient_id": self.recipient_id,
            "recipient_name": self.recipient_name,
            "subject": self.subject,
            "content": self.content,
            "conversation_id": self.conversation_id,
            "reply_to": self.reply_to,
            "requires_ack": self.requires_ack,
            "ttl": self.ttl,
        })
        return base

    def create_reply(
        self,
        content: Any,
        message_type: MessageType = MessageType.RESPONSE,
    ) -> "Message":
        """Create a reply to this message."""
        return Message(
            message_type=message_type,
            sender_id=self.recipient_id,
            sender_name=self.recipient_name,
            recipient_id=self.sender_id,
            recipient_name=self.sender_name,
            subject=f"Re: {self.subject}",
            content=content,
            conversation_id=self.conversation_id or self.event_id,
            reply_to=self.event_id,
        )
