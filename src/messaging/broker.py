from __future__ import annotations
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Any, TYPE_CHECKING

import redis.asyncio as redis

from .messages import Message, MessageType, Envelope

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    redis_url: str = "redis://localhost:6379"
    channel_prefix: str = "agents"
    broadcast_channel: str = "broadcast"
    dead_letter_channel: str = "dead_letters"
    max_retries: int = 3
    ack_timeout: float = 5.0


class MessageBroker:
    """
    Redis-based message broker for multi-agent communication.
    Supports:
    - Direct messaging (point-to-point)
    - Broadcast (all agents)
    - Topics/channels (pub/sub groups)
    - Message persistence and replay
    - Dead letter handling
    """

    def __init__(self, config: Optional[BrokerConfig] = None):
        self.config = config or BrokerConfig()
        self._redis: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None

        # Local agent registry (for same-process agents)
        self._local_agents: dict[str, Agent] = {}

        # Topic subscriptions: topic -> set of agent_ids
        self._topic_subscriptions: dict[str, set[str]] = {}

        # Message handlers by channel
        self._handlers: dict[str, Callable] = {}

        # Running state
        self._running = False
        self._listener_task: Optional[asyncio.Task] = None

    async def connect(self):
        """Connect to Redis."""
        self._redis = redis.from_url(self.config.redis_url)
        self._pubsub = self._redis.pubsub()
        logger.info(f"Connected to Redis at {self.config.redis_url}")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()
        logger.info("Disconnected from Redis")

    def _channel_for_agent(self, agent_id: str) -> str:
        return f"{self.config.channel_prefix}:{agent_id}"

    def _channel_for_topic(self, topic: str) -> str:
        return f"{self.config.channel_prefix}:topic:{topic}"

    async def register_agent(self, agent: Agent):
        """Register an agent with the broker."""
        self._local_agents[agent.agent_id] = agent

        # Subscribe to agent's direct channel
        channel = self._channel_for_agent(agent.agent_id)
        await self._pubsub.subscribe(channel)

        # Subscribe to broadcast channel
        broadcast = f"{self.config.channel_prefix}:{self.config.broadcast_channel}"
        await self._pubsub.subscribe(broadcast)

        # Store agent info in Redis for discovery
        agent_key = f"{self.config.channel_prefix}:registry:{agent.agent_id}"
        await self._redis.hset(agent_key, mapping={
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "status": "online",
            "registered_at": datetime.now(timezone.utc).isoformat(),
        })
        await self._redis.expire(agent_key, 300)  # 5 min TTL, refresh with heartbeat

        logger.info(f"Registered agent {agent.agent_id}")

    async def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self._local_agents:
            del self._local_agents[agent_id]

        channel = self._channel_for_agent(agent_id)
        await self._pubsub.unsubscribe(channel)

        agent_key = f"{self.config.channel_prefix}:registry:{agent_id}"
        await self._redis.delete(agent_key)

        logger.info(f"Unregistered agent {agent_id}")

    async def subscribe_topic(self, agent_id: str, topic: str):
        """Subscribe an agent to a topic."""
        if topic not in self._topic_subscriptions:
            self._topic_subscriptions[topic] = set()
            # Subscribe to topic channel
            channel = self._channel_for_topic(topic)
            await self._pubsub.subscribe(channel)

        self._topic_subscriptions[topic].add(agent_id)
        logger.info(f"Agent {agent_id} subscribed to topic {topic}")

    async def unsubscribe_topic(self, agent_id: str, topic: str):
        """Unsubscribe an agent from a topic."""
        if topic in self._topic_subscriptions:
            self._topic_subscriptions[topic].discard(agent_id)
            if not self._topic_subscriptions[topic]:
                channel = self._channel_for_topic(topic)
                await self._pubsub.unsubscribe(channel)
                del self._topic_subscriptions[topic]

    async def route(self, envelope: Envelope):
        """Route a message to its destination(s)."""
        msg = envelope.message
        data = envelope.to_json()

        if msg.recipient_id == "*":
            # Broadcast
            channel = f"{self.config.channel_prefix}:{self.config.broadcast_channel}"
            await self._redis.publish(channel, data)
            logger.debug(f"Broadcast message {msg.msg_id}")

        elif msg.recipient_id.startswith("topic:"):
            # Topic multicast
            topic = msg.recipient_id[6:]
            channel = self._channel_for_topic(topic)
            await self._redis.publish(channel, data)
            logger.debug(f"Multicast message {msg.msg_id} to topic {topic}")

        else:
            # Direct message
            # First try local delivery
            if msg.recipient_id in self._local_agents:
                agent = self._local_agents[msg.recipient_id]
                await agent.inbound.deliver(envelope)
                logger.debug(f"Local delivery of {msg.msg_id} to {msg.recipient_id}")
            else:
                # Remote delivery via Redis
                channel = self._channel_for_agent(msg.recipient_id)
                await self._redis.publish(channel, data)
                logger.debug(f"Remote delivery of {msg.msg_id} to {msg.recipient_id}")

        # Persist message for replay/audit
        await self._persist_message(envelope)

    async def _persist_message(self, envelope: Envelope):
        """Store message in Redis stream for persistence."""
        msg = envelope.message
        stream_key = f"{self.config.channel_prefix}:stream:{msg.recipient_id}"

        await self._redis.xadd(
            stream_key,
            {
                "msg_id": msg.msg_id,
                "data": envelope.to_json(),
            },
            maxlen=10000,  # Keep last 10k messages
        )

    async def get_message_history(
        self,
        agent_id: str,
        count: int = 100,
        start_id: str = "-",
    ) -> list[Envelope]:
        """Get message history for an agent."""
        stream_key = f"{self.config.channel_prefix}:stream:{agent_id}"
        messages = await self._redis.xrange(stream_key, start_id, "+", count=count)

        result = []
        for msg_id, data in messages:
            try:
                envelope = Envelope.from_json(data[b"data"].decode())
                result.append(envelope)
            except Exception as e:
                logger.error(f"Failed to parse message {msg_id}: {e}")

        return result

    async def start(self):
        """Start the message listener."""
        self._running = True
        self._listener_task = asyncio.create_task(self._listener_loop())
        logger.info("Message broker started")

    async def stop(self):
        """Stop the message listener."""
        self._running = False
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
        logger.info("Message broker stopped")

    async def _listener_loop(self):
        """Listen for incoming messages."""
        while self._running:
            try:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.1,
                )
                if message and message["type"] == "message":
                    await self._handle_message(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Listener error: {e}")
                await asyncio.sleep(0.1)

    async def _handle_message(self, redis_msg: dict):
        """Handle an incoming Redis message."""
        try:
            channel = redis_msg["channel"].decode()
            data = redis_msg["data"].decode()
            envelope = Envelope.from_json(data)
            msg = envelope.message

            # Determine target agents
            target_agents = []

            if self.config.broadcast_channel in channel:
                # Broadcast - deliver to all local agents
                target_agents = list(self._local_agents.values())
            elif ":topic:" in channel:
                # Topic - deliver to subscribed agents
                topic = channel.split(":topic:")[-1]
                if topic in self._topic_subscriptions:
                    for agent_id in self._topic_subscriptions[topic]:
                        if agent_id in self._local_agents:
                            target_agents.append(self._local_agents[agent_id])
            else:
                # Direct - extract agent_id from channel
                agent_id = channel.split(":")[-1]
                if agent_id in self._local_agents:
                    target_agents.append(self._local_agents[agent_id])

            # Deliver to targets
            for agent in target_agents:
                await agent.inbound.deliver(envelope)

        except Exception as e:
            logger.error(f"Failed to handle message: {e}")

    async def discover_agents(self, pattern: str = "*") -> list[dict]:
        """Discover registered agents."""
        key_pattern = f"{self.config.channel_prefix}:registry:{pattern}"
        keys = await self._redis.keys(key_pattern)

        agents = []
        for key in keys:
            data = await self._redis.hgetall(key)
            if data:
                agents.append({k.decode(): v.decode() for k, v in data.items()})

        return agents

    async def send_heartbeat(self, agent_id: str):
        """Refresh agent registration TTL."""
        agent_key = f"{self.config.channel_prefix}:registry:{agent_id}"
        await self._redis.hset(agent_key, "last_heartbeat", datetime.now(timezone.utc).isoformat())
        await self._redis.expire(agent_key, 300)
