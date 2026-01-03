from __future__ import annotations
import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Optional, Any

from .messages import Message, MessageType, Priority
from .mailbox import InboundMailbox, OutboundMailbox
from .broker import MessageBroker

logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


@dataclass
class AgentConfig:
    agent_id: Optional[str] = None
    agent_type: str = "generic"
    max_inbox_size: int = 10000
    rate_limit: int = 1000
    heartbeat_interval: float = 30.0
    system_prompt: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Agent(ABC):
    """
    Base class for all agents in the multi-agent system.
    Provides messaging infrastructure and lifecycle management.
    """

    def __init__(self, broker: MessageBroker, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.agent_id = self.config.agent_id or f"{self.config.agent_type}_{uuid.uuid4().hex[:8]}"
        self.agent_type = self.config.agent_type

        self.broker = broker
        self.status = AgentStatus.INITIALIZING

        # Mailboxes
        self.inbound = InboundMailbox(
            agent_id=self.agent_id,
            max_size=self.config.max_inbox_size,
        )
        self.outbound = OutboundMailbox(
            agent_id=self.agent_id,
            broker=broker,
            rate_limit=self.config.rate_limit,
        )

        # Lifecycle
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._created_at = datetime.utcnow()

        # Register default message handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register handlers for system messages."""
        self.inbound.register_handler(MessageType.HEARTBEAT, self._handle_heartbeat)
        self.inbound.register_handler(MessageType.ACK, self._handle_ack)
        self.inbound.register_handler(MessageType.TERMINATE, self._handle_terminate)
        self.inbound.register_handler(MessageType.SUSPEND, self._handle_suspend)
        self.inbound.register_handler(MessageType.RESUME, self._handle_resume)

    async def _handle_heartbeat(self, msg: Message):
        """Respond to heartbeat."""
        await self.send(
            msg.sender_id,
            MessageType.ACK,
            {"status": self.status.value, "timestamp": datetime.utcnow().isoformat()},
            correlation_id=msg.msg_id,
        )

    async def _handle_ack(self, msg: Message):
        """Handle acknowledgment."""
        if msg.correlation_id:
            self.outbound.receive_ack(msg.correlation_id)

    async def _handle_terminate(self, msg: Message):
        """Handle termination request."""
        logger.info(f"Agent {self.agent_id} received termination request")
        await self.stop()

    async def _handle_suspend(self, msg: Message):
        """Handle suspend request."""
        self.status = AgentStatus.SUSPENDED
        logger.info(f"Agent {self.agent_id} suspended")

    async def _handle_resume(self, msg: Message):
        """Handle resume request."""
        if self.status == AgentStatus.SUSPENDED:
            self.status = AgentStatus.RUNNING
            logger.info(f"Agent {self.agent_id} resumed")

    async def start(self):
        """Start the agent."""
        logger.info(f"Starting agent {self.agent_id}")

        # Register with broker
        await self.broker.register_agent(self)

        # Start outbound sender
        await self.outbound.start()

        # Start processing loop
        self._running = True
        self.status = AgentStatus.RUNNING

        # Start background tasks
        self._tasks.append(asyncio.create_task(self._message_loop()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

        # Call subclass initialization
        await self.on_start()

        logger.info(f"Agent {self.agent_id} started")

    async def stop(self):
        """Stop the agent."""
        logger.info(f"Stopping agent {self.agent_id}")
        self.status = AgentStatus.TERMINATING
        self._running = False

        # Call subclass cleanup
        await self.on_stop()

        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop outbound
        await self.outbound.stop()

        # Unregister from broker
        await self.broker.unregister_agent(self.agent_id)

        self.status = AgentStatus.TERMINATED
        logger.info(f"Agent {self.agent_id} stopped")

    async def _message_loop(self):
        """Main message processing loop."""
        while self._running:
            if self.status == AgentStatus.SUSPENDED:
                await asyncio.sleep(0.1)
                continue

            try:
                processed = await self.inbound.process_one()
                if not processed:
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in message loop: {e}")
                await asyncio.sleep(0.1)

    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval)
            await self.broker.send_heartbeat(self.agent_id)

    # Convenience methods for sending
    async def send(
        self,
        recipient_id: str,
        msg_type: MessageType,
        payload: dict,
        priority: Priority = Priority.NORMAL,
        **kwargs,
    ) -> str:
        """Send a message to another agent."""
        return await self.outbound.send(
            recipient_id, msg_type, payload, priority, **kwargs
        )

    async def broadcast(
        self,
        msg_type: MessageType,
        payload: dict,
        priority: Priority = Priority.NORMAL,
    ) -> str:
        """Broadcast a message to all agents."""
        return await self.outbound.broadcast(msg_type, payload, priority)

    async def request(
        self,
        recipient_id: str,
        payload: dict,
        timeout: float = 30.0,
    ) -> Optional[Message]:
        """Send a request and wait for response."""
        msg_id = await self.send(
            recipient_id,
            MessageType.REQUEST,
            payload,
            reply_to=self.agent_id,
        )

        # Wait for response with matching correlation_id
        start = datetime.utcnow()
        while (datetime.utcnow() - start).total_seconds() < timeout:
            envelope = await self.inbound.receive(timeout=0.1)
            if envelope:
                msg = envelope.message
                if msg.correlation_id == msg_id and msg.msg_type == MessageType.RESPONSE:
                    return msg
                # Put back if not our response
                await self.inbound.deliver(envelope)
            await asyncio.sleep(0.05)

        return None

    async def subscribe(self, topic: str):
        """Subscribe to a topic."""
        await self.broker.subscribe_topic(self.agent_id, topic)

    async def unsubscribe(self, topic: str):
        """Unsubscribe from a topic."""
        await self.broker.unsubscribe_topic(self.agent_id, topic)

    async def publish(self, topic: str, payload: dict, priority: Priority = Priority.NORMAL):
        """Publish to a topic."""
        return await self.send(f"topic:{topic}", MessageType.MULTICAST, payload, priority)

    # Abstract methods for subclasses
    @abstractmethod
    async def on_start(self):
        """Called when agent starts. Override to initialize."""
        pass

    @abstractmethod
    async def on_stop(self):
        """Called when agent stops. Override to cleanup."""
        pass

    @abstractmethod
    async def on_message(self, msg: Message):
        """Called for custom message handling. Override to process."""
        pass


class AgentRegistry:
    """
    Registry for managing multiple agents.
    Provides discovery, lifecycle management, and coordination.
    """

    def __init__(self, broker: MessageBroker):
        self.broker = broker
        self._agents: dict[str, Agent] = {}
        self._agent_types: dict[str, type] = {}

    def register_type(self, agent_type: str, agent_class: type):
        """Register an agent type for spawning."""
        self._agent_types[agent_type] = agent_class

    async def spawn(
        self,
        agent_type: str,
        config: Optional[AgentConfig] = None,
        **kwargs,
    ) -> Agent:
        """Spawn a new agent of the given type."""
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")

        agent_class = self._agent_types[agent_type]
        config = config or AgentConfig(agent_type=agent_type)

        agent = agent_class(self.broker, config, **kwargs)
        await agent.start()

        self._agents[agent.agent_id] = agent
        return agent

    async def terminate(self, agent_id: str):
        """Terminate an agent."""
        if agent_id in self._agents:
            agent = self._agents[agent_id]
            await agent.stop()
            del self._agents[agent_id]

    async def terminate_all(self):
        """Terminate all agents."""
        for agent_id in list(self._agents.keys()):
            await self.terminate(agent_id)

    def get(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> list[dict]:
        """List all local agents."""
        return [
            {
                "agent_id": a.agent_id,
                "agent_type": a.agent_type,
                "status": a.status.value,
            }
            for a in self._agents.values()
        ]

    async def discover_remote(self, pattern: str = "*") -> list[dict]:
        """Discover all registered agents (local and remote)."""
        return await self.broker.discover_agents(pattern)
