from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MessageType(str, Enum):
    # Core message types
    DIRECT = "direct"           # Point-to-point
    BROADCAST = "broadcast"     # To all agents
    MULTICAST = "multicast"     # To a group/topic

    # System messages
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    NACK = "nack"

    # Agent lifecycle
    SPAWN = "spawn"
    TERMINATE = "terminate"
    SUSPEND = "suspend"
    RESUME = "resume"

    # Coordination
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATE = "delegate"
    REPORT = "report"

    # Consensus/voting
    PROPOSE = "propose"
    VOTE = "vote"
    COMMIT = "commit"
    ROLLBACK = "rollback"


class Priority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Message:
    sender_id: str
    recipient_id: str  # Can be agent_id, topic, or "*" for broadcast
    msg_type: MessageType
    payload: dict[str, Any]

    # Auto-generated
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Optional metadata
    priority: Priority = Priority.NORMAL
    ttl_seconds: Optional[int] = None  # Time to live
    correlation_id: Optional[str] = None  # For request/response pairing
    reply_to: Optional[str] = None  # Where to send response
    trace_id: Optional[str] = None  # Distributed tracing

    def to_json(self) -> str:
        d = asdict(self)
        d["msg_type"] = self.msg_type.value
        d["priority"] = self.priority.value
        return json.dumps(d)

    @classmethod
    def from_json(cls, data: str) -> Message:
        d = json.loads(data)
        d["msg_type"] = MessageType(d["msg_type"])
        d["priority"] = Priority(d["priority"])
        return cls(**d)

    def reply(self, payload: dict, msg_type: MessageType = MessageType.RESPONSE) -> Message:
        """Create a reply message"""
        return Message(
            sender_id=self.recipient_id,
            recipient_id=self.sender_id,
            msg_type=msg_type,
            payload=payload,
            correlation_id=self.msg_id,
            reply_to=self.reply_to,
            trace_id=self.trace_id,
        )


@dataclass
class Envelope:
    """Wrapper for routing metadata"""
    message: Message
    route: list[str] = field(default_factory=list)  # Hops taken
    retry_count: int = 0
    max_retries: int = 3

    def to_json(self) -> str:
        return json.dumps({
            "message": json.loads(self.message.to_json()),
            "route": self.route,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        })

    @classmethod
    def from_json(cls, data: str) -> Envelope:
        d = json.loads(data)
        return cls(
            message=Message.from_json(json.dumps(d["message"])),
            route=d["route"],
            retry_count=d["retry_count"],
            max_retries=d["max_retries"],
        )
