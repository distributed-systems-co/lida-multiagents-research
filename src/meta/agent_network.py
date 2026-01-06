"""Multi-agent network with Redis pub/sub and temporal graph persistence.

Enables long-running conversations between personality-driven agents with:
- Redis pub/sub for real-time message passing
- Temporal hypergraph for conversation state & history
- Personality evolution tracking over time
- Dynamic agent spawning and routing
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, List, Any, Callable, AsyncIterator

from .personality import Personality, PERSONALITY_ARCHETYPES
from .temporal import (
    get_temporal_graph,
    DynamicsType,
    TemporalEvent,
)
from .hypergraph import HyperNode
from .locality import (
    get_locality_network,
    Message,
    MessagePriority,
    DeliveryMode,
)


class AgentRole(str, Enum):
    """Role types for agents."""
    CONVERSANT = "conversant"      # Regular conversation participant
    MODERATOR = "moderator"        # Monitors and guides conversations
    OBSERVER = "observer"          # Watches but doesn't participate
    ORCHESTRATOR = "orchestrator"  # Manages agent lifecycle


@dataclass
class ConversationContext:
    """Context for an ongoing conversation."""
    conversation_id: str
    topic: str
    participants: List[str]  # Agent IDs
    messages: List[Dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    turn_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, agent_id: str, content: str) -> Dict[str, Any]:
        """Add a message to the conversation."""
        msg = {
            "id": str(uuid.uuid4()),
            "agent_id": agent_id,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "turn": self.turn_count,
        }
        self.messages.append(msg)
        self.turn_count += 1
        return msg

    def get_recent_context(self, max_messages: int = 5) -> str:
        """Get recent messages as context string."""
        recent = self.messages[-max_messages:]
        lines = []
        for m in recent:
            lines.append(f"{m['agent_id']}: {m['content']}")
        return "\n".join(lines)


@dataclass
class PersonalityAgent:
    """An agent with a personality that can participate in conversations."""
    agent_id: str
    personality: Personality
    role: AgentRole = AgentRole.CONVERSANT
    model_config: Optional[Any] = None  # MLXModelConfig or similar
    _client: Optional[Any] = None  # MLXClient or similar
    _redis_client: Optional[Any] = None
    _subscriptions: Dict[str, asyncio.Task] = field(default_factory=dict)
    _message_handlers: List[Callable] = field(default_factory=list)
    conversation_memory: Dict[str, ConversationContext] = field(default_factory=dict)

    def __post_init__(self):
        self.temporal_node_id = f"agent:{self.agent_id}"

    async def connect_redis(self, redis_url: str = "redis://localhost:6379"):
        """Connect to Redis for pub/sub."""
        try:
            import redis.asyncio as redis
            self._redis_client = redis.from_url(redis_url)
            await self._redis_client.ping()
            return True
        except Exception as e:
            print(f"Redis connection failed: {e}")
            return False

    async def subscribe(self, channel: str, handler: Optional[Callable] = None):
        """Subscribe to a Redis channel."""
        if not self._redis_client:
            raise RuntimeError("Not connected to Redis")

        pubsub = self._redis_client.pubsub()
        await pubsub.subscribe(channel)

        async def listener():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    if handler:
                        await handler(data)
                    for h in self._message_handlers:
                        await h(channel, data)

        task = asyncio.create_task(listener())
        self._subscriptions[channel] = task
        return task

    async def publish(self, channel: str, message: Dict[str, Any]):
        """Publish a message to a Redis channel."""
        if not self._redis_client:
            raise RuntimeError("Not connected to Redis")

        message["sender"] = self.agent_id
        message["timestamp"] = datetime.now(timezone.utc).isoformat()
        await self._redis_client.publish(channel, json.dumps(message))

    async def unsubscribe(self, channel: str):
        """Unsubscribe from a channel."""
        if channel in self._subscriptions:
            self._subscriptions[channel].cancel()
            del self._subscriptions[channel]

    def init_model(self):
        """Initialize the LLM client with personality."""
        if self.model_config is None:
            from .personality_local import MLXModelConfig
            self.model_config = MLXModelConfig(
                model_path="mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit",
                max_tokens=200,
                temperature=0.7,
            )

        from .personality_local import MLXClient
        self._client = MLXClient(self.model_config)
        self._client.personality = self.personality

    async def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a response using the personality-infused model."""
        if self._client is None:
            self.init_model()

        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\n{prompt}"

        result = await self._client.generate(full_prompt)
        return result.text.strip()

    def record_event(self, event_type: str, data: Dict[str, Any]):
        """Record an event to the temporal graph."""
        temporal = get_temporal_graph()
        # Ensure node exists
        if not temporal.get_node(self.temporal_node_id):
            node = HyperNode(
                node_id=self.temporal_node_id,
                node_type="agent",
                name=self.personality.name,
                metadata={
                    "personality": self.personality.name,
                },
            )
            temporal.add_node(node)
        # Emit event
        event = TemporalEvent(
            event_id=str(uuid.uuid4()),
            event_type=DynamicsType.ACTIVATION,  # Generic event type
            source_id=self.temporal_node_id,
            data={"type": event_type, **data},
        )
        temporal.emit_event(event)
        return event.event_id


class AgentNetwork:
    """Network of personality agents communicating via Redis pub/sub."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.agents: Dict[str, PersonalityAgent] = {}
        self.conversations: Dict[str, ConversationContext] = {}
        self._redis_client = None
        self._running = False

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            import redis.asyncio as redis
            self._redis_client = redis.from_url(self.redis_url)
            await self._redis_client.ping()
            print(f"Connected to Redis at {self.redis_url}")
            return True
        except Exception as e:
            print(f"Redis connection failed: {e} - running in local mode")
            return False

    def spawn_agent(
        self,
        personality_key: str,
        agent_id: Optional[str] = None,
        role: AgentRole = AgentRole.CONVERSANT,
        model_config: Optional[Any] = None,
    ) -> PersonalityAgent:
        """Spawn a new agent with a personality."""
        if personality_key not in PERSONALITY_ARCHETYPES:
            raise ValueError(f"Unknown personality: {personality_key}")

        personality = PERSONALITY_ARCHETYPES[personality_key]()
        agent_id = agent_id or f"{personality_key}_{uuid.uuid4().hex[:8]}"

        agent = PersonalityAgent(
            agent_id=agent_id,
            personality=personality,
            role=role,
            model_config=model_config,
        )

        self.agents[agent_id] = agent

        # Record in temporal graph
        temporal = get_temporal_graph()
        node = HyperNode(
            node_id=agent.temporal_node_id,
            node_type="agent",
            name=personality.name,
            metadata={
                "personality": personality_key,
                "role": role.value,
                "spawned_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        temporal.add_node(node)

        return agent

    async def create_conversation(
        self,
        topic: str,
        agent_ids: List[str],
        conversation_id: Optional[str] = None,
    ) -> ConversationContext:
        """Create a new conversation between agents."""
        conversation_id = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

        # Verify all agents exist
        for aid in agent_ids:
            if aid not in self.agents:
                raise ValueError(f"Unknown agent: {aid}")

        context = ConversationContext(
            conversation_id=conversation_id,
            topic=topic,
            participants=agent_ids,
        )

        self.conversations[conversation_id] = context

        # Record in temporal graph
        temporal = get_temporal_graph()
        conv_node_id = f"conversation:{conversation_id}"
        node = HyperNode(
            node_id=conv_node_id,
            node_type="conversation",
            name=f"Conversation: {topic[:30]}...",
            metadata={
                "topic": topic,
                "participants": agent_ids,
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        temporal.add_node(node)
        event = TemporalEvent(
            event_id=str(uuid.uuid4()),
            event_type=DynamicsType.ACTIVATION,
            source_id=conv_node_id,
            data={
                "type": "conversation_started",
                "topic": topic,
                "participants": agent_ids,
            },
        )
        temporal.emit_event(event)

        # Create Redis channel if connected
        if self._redis_client:
            channel = f"conv:{conversation_id}"
            for aid in agent_ids:
                agent = self.agents[aid]
                if agent._redis_client is None:
                    await agent.connect_redis(self.redis_url)
                await agent.subscribe(channel)

        return context

    async def run_conversation(
        self,
        conversation_id: str,
        turns: int = 10,
        delay: float = 0.5,
    ) -> ConversationContext:
        """Run a conversation for a specified number of turns."""
        if conversation_id not in self.conversations:
            raise ValueError(f"Unknown conversation: {conversation_id}")

        context = self.conversations[conversation_id]
        agents = [self.agents[aid] for aid in context.participants]

        # Initialize models
        for agent in agents:
            if agent._client is None:
                agent.init_model()

        # First agent starts
        current_idx = 0
        starter_prompt = f"""Topic: {context.topic}

Give your honest perspective in 2-3 sentences. Be true to your personality."""

        response = await agents[current_idx].generate_response(starter_prompt)
        msg = context.add_message(agents[current_idx].agent_id, response)

        # Record to temporal graph
        agents[current_idx].record_event("message_sent", {
            "conversation_id": conversation_id,
            "content": response,
            "turn": 0,
        })

        # Publish to Redis if connected
        channel = f"conv:{conversation_id}"
        if agents[current_idx]._redis_client:
            await agents[current_idx].publish(channel, {
                "type": "message",
                "agent_id": agents[current_idx].agent_id,
                "content": response,
                "turn": 0,
            })

        yield msg  # Yield first message

        # Continue conversation
        for turn in range(1, turns):
            await asyncio.sleep(delay)

            # Next agent responds
            current_idx = (current_idx + 1) % len(agents)
            agent = agents[current_idx]

            # Get previous message
            prev_msg = context.messages[-1]
            prev_agent = self.agents[prev_msg["agent_id"]]

            prompt = f"""You're discussing: {context.topic}

{prev_agent.personality.name} said:
"{prev_msg['content']}"

Respond directly in 2-3 sentences. Be true to your personality."""

            recent_context = context.get_recent_context(max_messages=4)
            response = await agent.generate_response(prompt, recent_context if turn > 2 else None)

            # Clean response
            import re
            response = re.sub(r'<\|[^>]+\|>', '', response)
            response = response.strip()

            msg = context.add_message(agent.agent_id, response)

            # Record to temporal graph
            agent.record_event("message_sent", {
                "conversation_id": conversation_id,
                "content": response,
                "turn": turn,
            })

            # Publish to Redis
            if agent._redis_client:
                await agent.publish(channel, {
                    "type": "message",
                    "agent_id": agent.agent_id,
                    "content": response,
                    "turn": turn,
                })

            yield msg

        # Mark conversation complete
        context.metadata["completed"] = True
        context.metadata["ended_at"] = datetime.now(timezone.utc).isoformat()

        temporal = get_temporal_graph()
        event = TemporalEvent(
            event_id=str(uuid.uuid4()),
            event_type=DynamicsType.DECAY,  # Conversation ending
            source_id=f"conversation:{conversation_id}",
            data={
                "type": "conversation_ended",
                "turns": turns,
                "messages": len(context.messages),
            },
        )
        temporal.emit_event(event)

    async def run_parallel_conversations(
        self,
        conversations: List[Dict[str, Any]],
        turns: int = 5,
    ):
        """Run multiple conversations in parallel."""
        tasks = []

        for conv_config in conversations:
            # Spawn agents if needed
            agent_ids = []
            for personality in conv_config["personalities"]:
                agent = self.spawn_agent(personality)
                agent_ids.append(agent.agent_id)

            # Create conversation
            context = await self.create_conversation(
                topic=conv_config["topic"],
                agent_ids=agent_ids,
            )

            # Create task
            async def run_and_collect(conv_id, t):
                messages = []
                async for msg in self.run_conversation(conv_id, t):
                    messages.append(msg)
                return messages

            tasks.append(run_and_collect(context.conversation_id, turns))

        # Run all conversations in parallel
        results = await asyncio.gather(*tasks)
        return results

    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history from temporal graph."""
        temporal = get_temporal_graph()
        conv_node = f"conversation:{conversation_id}"

        # Query events for this conversation
        events = temporal.query_events(source=conv_node)

        messages = []
        for event in events:
            if hasattr(event, 'event_type') and event.event_type == "message_sent":
                messages.append(event.data if hasattr(event, 'data') else {})

        return sorted(messages, key=lambda x: x.get("turn", 0))

    async def cleanup(self):
        """Cleanup connections."""
        for agent in self.agents.values():
            for task in agent._subscriptions.values():
                task.cancel()
            if agent._redis_client:
                await agent._redis_client.close()

        if self._redis_client:
            await self._redis_client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────

_network: Optional[AgentNetwork] = None

def get_agent_network(redis_url: str = "redis://localhost:6379") -> AgentNetwork:
    """Get or create the global agent network."""
    global _network
    if _network is None:
        _network = AgentNetwork(redis_url)
    return _network

def reset_agent_network():
    """Reset the global agent network."""
    global _network
    _network = None


async def quick_conversation(
    personality1: str,
    personality2: str,
    topic: str,
    turns: int = 6,
    redis_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Quick function to run a conversation between two personalities."""
    network = AgentNetwork(redis_url) if redis_url else AgentNetwork()

    if redis_url:
        await network.connect()

    agent1 = network.spawn_agent(personality1)
    agent2 = network.spawn_agent(personality2)

    context = await network.create_conversation(
        topic=topic,
        agent_ids=[agent1.agent_id, agent2.agent_id],
    )

    messages = []
    async for msg in network.run_conversation(context.conversation_id, turns):
        messages.append(msg)
        print(f"  [{msg['agent_id']}]: {msg['content']}\n")

    await network.cleanup()
    return messages
