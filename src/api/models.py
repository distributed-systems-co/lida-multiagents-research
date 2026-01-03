"""Pydantic models for API requests/responses."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentType(str, Enum):
    DEMIURGE = "demiurge"
    PERSONA = "persona"
    WORKER = "worker"


class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    TERMINATED = "terminated"
    ERROR = "error"
    DEAD = "dead"


# Request models
class SpawnAgentRequest(BaseModel):
    agent_type: AgentType
    agent_id: Optional[str] = None
    config: dict = Field(default_factory=dict)


class ChatMessageRequest(BaseModel):
    agent_id: str
    message: str
    stream: bool = False
    model: Optional[str] = None  # Override default model
    use_llm: bool = True  # Whether to use LLM for response


class StreamingChatRequest(BaseModel):
    agent_id: str
    message: str
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096


class SignatureRequest(BaseModel):
    """Create a dynamic signature for structured LLM calls."""
    name: str
    description: Optional[str] = None
    instructions: Optional[str] = None
    inputs: list[dict] = Field(default_factory=list)  # [{name, description, type, required}]
    outputs: list[dict] = Field(default_factory=list)


class ExecuteSignatureRequest(BaseModel):
    """Execute a signature with inputs."""
    signature_name: str  # Predefined or custom signature
    inputs: dict
    model: Optional[str] = None
    stream: bool = False


class SendMessageRequest(BaseModel):
    sender_id: str
    recipient_id: str
    message_type: str = "REQUEST"
    payload: dict = Field(default_factory=dict)


class CreateDatasetRequest(BaseModel):
    name: str
    description: Optional[str] = None
    schema_def: Optional[dict] = None
    tags: list[str] = Field(default_factory=list)


class AddDataRequest(BaseModel):
    dataset_id: str
    records: list[dict]


# Response models
class AgentInfo(BaseModel):
    agent_id: str
    agent_type: str
    status: str
    inbox_count: int = 0
    outbox_count: int = 0
    processed_count: int = 0
    start_count: int = 0
    last_start: Optional[datetime] = None
    last_error: Optional[str] = None


class MessageInfo(BaseModel):
    msg_id: str
    sender_id: str
    recipient_id: str
    msg_type: str
    timestamp: datetime
    payload: dict


class DatasetInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    record_count: int = 0
    created_at: datetime
    updated_at: datetime
    tags: list[str] = Field(default_factory=list)
    schema_def: Optional[dict] = None


class StatsInfo(BaseModel):
    total_agents: int
    running_agents: int
    messages_sent: int
    messages_received: int
    broadcasts: int
    multicasts: int
    direct_messages: int
    errors: int
    datasets: int
    uptime_seconds: float


class ChatResponse(BaseModel):
    agent_id: str
    response: str
    timestamp: datetime
