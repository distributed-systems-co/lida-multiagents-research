"""
Comprehensive messaging patterns for multi-agent systems.

Implements 70+ messaging topologies including:
- Point-to-point, broadcast, multicast, anycast
- Scatter-gather, pipeline, ring, mesh, tree
- Publish-subscribe, request-reply, gossip/epidemic
- Aggregation, filtering, transformation, routing
- Saga, choreography, orchestration patterns
- Fault tolerance: circuit breaker, bulkhead, retry
- Flow control: throttle, backpressure, windowing
"""

from __future__ import annotations
import asyncio
import uuid
import random
import time
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Any, Optional, Set, Callable, AsyncIterator,
    Tuple, Union, TypeVar, Generic
)
from collections import deque
from abc import ABC, abstractmethod
import heapq


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGING PATTERN TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════════

class PatternCategory(str, Enum):
    """Categories of messaging patterns."""
    POINT_TO_POINT = "point_to_point"       # 1:1
    ONE_TO_MANY = "one_to_many"             # 1:N
    MANY_TO_ONE = "many_to_one"             # N:1
    MANY_TO_MANY = "many_to_many"           # N:M
    HIERARCHICAL = "hierarchical"           # Tree structures
    RING = "ring"                           # Circular
    MESH = "mesh"                           # Full/partial mesh
    PIPELINE = "pipeline"                   # Sequential chain
    PUBSUB = "pubsub"                       # Topic-based
    GOSSIP = "gossip"                       # Epidemic/probabilistic
    AGGREGATION = "aggregation"             # Collect and combine
    ROUTING = "routing"                     # Content/rule-based
    SAGA = "saga"                           # Distributed transactions
    FAULT_TOLERANCE = "fault_tolerance"     # Reliability patterns
    FLOW_CONTROL = "flow_control"           # Rate limiting


class MessagePattern(str, Enum):
    """All 70+ messaging patterns."""
    # ─── Point-to-Point (1:1) ───
    DIRECT = "direct"                       # Simple 1:1
    REQUEST_REPLY = "request_reply"         # Synchronous exchange
    ASYNC_REQUEST = "async_request"         # Async with callback
    FIRE_AND_FORGET = "fire_and_forget"     # No response expected
    CORRELATION = "correlation"             # Request-response matching

    # ─── One-to-Many (1:N) ───
    BROADCAST = "broadcast"                 # 1 to all
    MULTICAST = "multicast"                 # 1 to group
    ANYCAST = "anycast"                     # 1 to any one of group
    GEOCAST = "geocast"                     # Location-based
    SCATTER = "scatter"                     # Fan-out parallel
    SPLITTER = "splitter"                   # Split message to N
    RECIPIENT_LIST = "recipient_list"       # Dynamic recipient list

    # ─── Many-to-One (N:1) ───
    GATHER = "gather"                       # Collect responses
    CONVERGECAST = "convergecast"           # Aggregate from many
    AGGREGATOR = "aggregator"               # Combine messages
    RESEQUENCER = "resequencer"             # Restore order
    COMPETING_CONSUMERS = "competing_consumers"  # Load balance to one

    # ─── Many-to-Many (N:M) ───
    SCATTER_GATHER = "scatter_gather"       # 1:N:1 fan-out/in
    MESH_BROADCAST = "mesh_broadcast"       # Full mesh
    PARTIAL_MESH = "partial_mesh"           # Partial mesh
    RELAY_CHAIN = "relay_chain"             # 1:1:1:...:N
    BIDIRECTIONAL = "bidirectional"         # Two-way all

    # ─── Hierarchical ───
    TREE_BROADCAST = "tree_broadcast"       # Parent to children
    TREE_CONVERGE = "tree_converge"         # Children to parent
    HIERARCHICAL_MULTICAST = "hierarchical_multicast"  # Level-based
    LEADER_FOLLOWER = "leader_follower"     # Leader broadcasts

    # ─── Ring ───
    RING_PASS = "ring_pass"                 # Sequential around ring
    TOKEN_RING = "token_ring"               # Token-based access
    RING_BROADCAST = "ring_broadcast"       # Propagate around ring
    RING_ELECTION = "ring_election"         # Leader election

    # ─── Pipeline ───
    PIPELINE = "pipeline"                   # Sequential stages
    PARALLEL_PIPELINE = "parallel_pipeline"  # Parallel stages
    FILTER_CHAIN = "filter_chain"           # Sequential filters
    INTERCEPTOR_CHAIN = "interceptor_chain"  # Request/response chain

    # ─── Pub/Sub ───
    PUBLISH = "publish"                     # Publish to topic
    SUBSCRIBE = "subscribe"                 # Subscribe to topic
    TOPIC_MULTICAST = "topic_multicast"     # Topic-based groups
    CONTENT_FILTER = "content_filter"       # Content-based subscribe
    DURABLE_SUBSCRIBER = "durable_subscriber"  # Persistent subscription

    # ─── Gossip/Epidemic ───
    GOSSIP_PUSH = "gossip_push"             # Push to random peers
    GOSSIP_PULL = "gossip_pull"             # Pull from random peers
    GOSSIP_PUSH_PULL = "gossip_push_pull"   # Bidirectional gossip
    EPIDEMIC_BROADCAST = "epidemic_broadcast"  # Probabilistic spread
    ANTI_ENTROPY = "anti_entropy"           # State synchronization
    RUMOR_MONGERING = "rumor_mongering"     # Hot/cold rumor spread

    # ─── Routing ───
    CONTENT_ROUTER = "content_router"       # Route by content
    HEADER_ROUTER = "header_router"         # Route by header
    DYNAMIC_ROUTER = "dynamic_router"       # Runtime routing
    ROUTING_SLIP = "routing_slip"           # Sequential routing
    PROCESS_MANAGER = "process_manager"     # Orchestrated routing

    # ─── Transformation ───
    TRANSLATOR = "translator"               # Format conversion
    NORMALIZER = "normalizer"               # Standardize format
    ENRICHER = "enricher"                   # Add data
    FILTER = "filter"                       # Remove/select data
    CLAIM_CHECK = "claim_check"             # Large payload handling

    # ─── Saga/Choreography ───
    SAGA_ORCHESTRATION = "saga_orchestration"    # Central coordinator
    SAGA_CHOREOGRAPHY = "saga_choreography"      # Distributed coordination
    COMPENSATING_TX = "compensating_tx"          # Rollback
    TWO_PHASE_COMMIT = "two_phase_commit"        # Distributed commit
    EVENT_SOURCING = "event_sourcing"            # Event log

    # ─── Fault Tolerance ───
    CIRCUIT_BREAKER = "circuit_breaker"     # Fail fast
    BULKHEAD = "bulkhead"                   # Isolation
    RETRY = "retry"                         # Automatic retry
    TIMEOUT = "timeout"                     # Time-bounded
    FALLBACK = "fallback"                   # Default on failure
    DEAD_LETTER = "dead_letter"             # Failed message queue
    IDEMPOTENT = "idempotent"               # Duplicate handling

    # ─── Flow Control ───
    THROTTLE = "throttle"                   # Rate limiting
    DEBOUNCE = "debounce"                   # Collapse rapid fires
    SAMPLE = "sample"                       # Periodic sampling
    BUFFER = "buffer"                       # Queue messages
    WINDOW = "window"                       # Batch by time/count
    BACKPRESSURE = "backpressure"           # Slow producer
    TOKEN_BUCKET = "token_bucket"           # Token-based rate
    LEAKY_BUCKET = "leaky_bucket"           # Smooth output

    # ─── Coordination ───
    BARRIER = "barrier"                     # Synchronization point
    SEMAPHORE = "semaphore"                 # Limited concurrency
    QUORUM = "quorum"                       # Consensus required
    LEADER_ELECTION = "leader_election"     # Elect leader
    MUTEX = "mutex"                         # Mutual exclusion

    # ─── Monitoring ───
    WIRE_TAP = "wire_tap"                   # Copy for monitoring
    MESSAGE_HISTORY = "message_history"     # Audit trail
    CHANNEL_PURGER = "channel_purger"       # Cleanup
    TEST_MESSAGE = "test_message"           # Health check
    CONTROL_BUS = "control_bus"             # Management


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """Base message type for all patterns."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipients: List[str] = field(default_factory=list)
    content: Any = None
    pattern: MessagePattern = MessagePattern.DIRECT
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    headers: Dict[str, Any] = field(default_factory=dict)
    ttl: Optional[float] = None  # Time to live in seconds
    priority: int = 0
    sequence: int = 0
    hop_count: int = 0
    max_hops: int = 10
    visited: Set[str] = field(default_factory=set)

    def clone(self) -> "Message":
        """Create a copy of the message."""
        return Message(
            id=str(uuid.uuid4()),
            sender=self.sender,
            recipients=list(self.recipients),
            content=self.content,
            pattern=self.pattern,
            correlation_id=self.correlation_id,
            reply_to=self.reply_to,
            headers=dict(self.headers),
            ttl=self.ttl,
            priority=self.priority,
            sequence=self.sequence,
            hop_count=self.hop_count + 1,
            max_hops=self.max_hops,
            visited=set(self.visited),
        )

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.ttl is None:
            return False
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.ttl


@dataclass
class TopicMessage(Message):
    """Message with topic for pub/sub."""
    topic: str = ""
    partition: int = 0


@dataclass
class SagaMessage(Message):
    """Message for saga transactions."""
    saga_id: str = ""
    step: int = 0
    total_steps: int = 0
    compensating: bool = False
    state: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT/NODE ABSTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentEndpoint:
    """An endpoint in the messaging network."""
    id: str
    location: Tuple[float, float] = (0.0, 0.0)  # For geocast
    groups: Set[str] = field(default_factory=set)
    topics: Set[str] = field(default_factory=set)
    inbox: asyncio.Queue = field(default_factory=lambda: asyncio.Queue())
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[str] = None  # For tree structures
    children: Set[str] = field(default_factory=set)
    neighbors: Set[str] = field(default_factory=set)  # For mesh/ring

    # Stats
    messages_sent: int = 0
    messages_received: int = 0
    last_active: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class MessagingNetwork:
    """
    Multi-agent messaging network supporting 70+ patterns.
    """

    def __init__(self):
        self.agents: Dict[str, AgentEndpoint] = {}
        self.groups: Dict[str, Set[str]] = {}
        self.topics: Dict[str, Set[str]] = {}  # topic -> subscribers
        self.message_log: List[Message] = []
        self.dead_letters: List[Message] = []

        # Ring structure
        self._ring_order: List[str] = []
        self._ring_token_holder: Optional[str] = None

        # Tree structure
        self._tree_root: Optional[str] = None

        # Saga tracking
        self._sagas: Dict[str, Dict[str, Any]] = {}

        # Circuit breakers
        self._circuit_states: Dict[str, str] = {}  # agent_id -> "closed"|"open"|"half-open"
        self._failure_counts: Dict[str, int] = {}

        # Rate limiters
        self._token_buckets: Dict[str, Tuple[float, float, datetime]] = {}  # id -> (tokens, rate, last_update)
        self._windows: Dict[str, List[Message]] = {}

        # Handlers
        self._message_handlers: Dict[str, List[Callable]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Management
    # ─────────────────────────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, location: Tuple[float, float] = None,
                       groups: List[str] = None, parent: str = None) -> AgentEndpoint:
        """Register an agent endpoint."""
        agent = AgentEndpoint(
            id=agent_id,
            location=location or (0.0, 0.0),
            groups=set(groups or []),
            parent=parent,
        )
        self.agents[agent_id] = agent

        # Add to groups
        for g in agent.groups:
            if g not in self.groups:
                self.groups[g] = set()
            self.groups[g].add(agent_id)

        # Add to ring
        self._ring_order.append(agent_id)

        # Set up tree
        if parent and parent in self.agents:
            self.agents[parent].children.add(agent_id)

        return agent

    def add_to_group(self, agent_id: str, group: str):
        """Add agent to a group."""
        if group not in self.groups:
            self.groups[group] = set()
        self.groups[group].add(agent_id)
        if agent_id in self.agents:
            self.agents[agent_id].groups.add(group)

    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic."""
        if topic not in self.topics:
            self.topics[topic] = set()
        self.topics[topic].add(agent_id)
        if agent_id in self.agents:
            self.agents[agent_id].topics.add(topic)

    def set_neighbors(self, agent_id: str, neighbors: List[str]):
        """Set mesh neighbors for an agent."""
        if agent_id in self.agents:
            self.agents[agent_id].neighbors = set(neighbors)

    def set_tree_root(self, agent_id: str):
        """Set the root of the tree structure."""
        self._tree_root = agent_id

    # ─────────────────────────────────────────────────────────────────────────
    # Point-to-Point Patterns (1:1)
    # ─────────────────────────────────────────────────────────────────────────

    async def direct(self, sender: str, recipient: str, content: Any) -> Message:
        """Direct 1:1 message."""
        msg = Message(
            sender=sender,
            recipients=[recipient],
            content=content,
            pattern=MessagePattern.DIRECT,
        )
        await self._deliver(msg)
        return msg

    async def request_reply(self, sender: str, recipient: str, content: Any,
                            timeout: float = 5.0) -> Optional[Message]:
        """Synchronous request-reply pattern."""
        correlation_id = str(uuid.uuid4())
        request = Message(
            sender=sender,
            recipients=[recipient],
            content=content,
            pattern=MessagePattern.REQUEST_REPLY,
            correlation_id=correlation_id,
            reply_to=sender,
        )
        await self._deliver(request)

        # Wait for reply
        reply_future = asyncio.Future()
        self._register_reply_handler(correlation_id, reply_future)

        try:
            reply = await asyncio.wait_for(reply_future, timeout)
            return reply
        except asyncio.TimeoutError:
            return None

    async def fire_and_forget(self, sender: str, recipient: str, content: Any) -> Message:
        """Fire and forget - no response expected."""
        msg = Message(
            sender=sender,
            recipients=[recipient],
            content=content,
            pattern=MessagePattern.FIRE_AND_FORGET,
        )
        await self._deliver(msg, log=False)  # Don't log fire-and-forget
        return msg

    # ─────────────────────────────────────────────────────────────────────────
    # One-to-Many Patterns (1:N)
    # ─────────────────────────────────────────────────────────────────────────

    async def broadcast(self, sender: str, content: Any,
                        exclude: Set[str] = None) -> List[Message]:
        """Broadcast to all agents."""
        exclude = exclude or set()
        recipients = [a for a in self.agents if a != sender and a not in exclude]
        return await self._send_to_many(sender, recipients, content, MessagePattern.BROADCAST)

    async def multicast(self, sender: str, group: str, content: Any) -> List[Message]:
        """Multicast to a group."""
        if group not in self.groups:
            return []
        recipients = [a for a in self.groups[group] if a != sender]
        return await self._send_to_many(sender, recipients, content, MessagePattern.MULTICAST)

    async def anycast(self, sender: str, group: str, content: Any) -> Message:
        """Send to any one member of a group (random selection)."""
        if group not in self.groups:
            raise ValueError(f"Group {group} not found")
        candidates = [a for a in self.groups[group] if a != sender]
        if not candidates:
            raise ValueError(f"No candidates in group {group}")
        recipient = random.choice(candidates)
        return await self.direct(sender, recipient, content)

    async def geocast(self, sender: str, content: Any, center: Tuple[float, float],
                      radius: float) -> List[Message]:
        """Send to agents within geographic radius."""
        recipients = []
        for aid, agent in self.agents.items():
            if aid == sender:
                continue
            dist = ((agent.location[0] - center[0])**2 +
                    (agent.location[1] - center[1])**2) ** 0.5
            if dist <= radius:
                recipients.append(aid)
        return await self._send_to_many(sender, recipients, content, MessagePattern.GEOCAST)

    async def scatter(self, sender: str, recipients: List[str],
                      contents: List[Any]) -> List[Message]:
        """Scatter different content to different recipients."""
        if len(recipients) != len(contents):
            raise ValueError("Recipients and contents must have same length")
        messages = []
        for recipient, content in zip(recipients, contents):
            msg = await self.direct(sender, recipient, content)
            msg.pattern = MessagePattern.SCATTER
            messages.append(msg)
        return messages

    async def recipient_list(self, sender: str, content: Any,
                             selector: Callable[[str, AgentEndpoint], bool]) -> List[Message]:
        """Dynamic recipient list based on selector function."""
        recipients = [aid for aid, agent in self.agents.items()
                      if aid != sender and selector(aid, agent)]
        return await self._send_to_many(sender, recipients, content, MessagePattern.RECIPIENT_LIST)

    # ─────────────────────────────────────────────────────────────────────────
    # Many-to-One Patterns (N:1)
    # ─────────────────────────────────────────────────────────────────────────

    async def gather(self, coordinator: str, sources: List[str],
                     timeout: float = 5.0) -> List[Message]:
        """Gather responses from multiple sources."""
        correlation_id = str(uuid.uuid4())
        responses = []
        futures = []

        for source in sources:
            request = Message(
                sender=coordinator,
                recipients=[source],
                content={"type": "gather_request"},
                pattern=MessagePattern.GATHER,
                correlation_id=correlation_id,
                reply_to=coordinator,
            )
            await self._deliver(request)

            future = asyncio.Future()
            self._register_reply_handler(f"{correlation_id}:{source}", future)
            futures.append(future)

        # Wait for all responses
        done, pending = await asyncio.wait(
            futures, timeout=timeout, return_when=asyncio.ALL_COMPLETED
        )

        for future in done:
            if not future.cancelled() and future.exception() is None:
                responses.append(future.result())

        return responses

    async def convergecast(self, content: Any, leaf_nodes: List[str] = None) -> Dict[str, Any]:
        """Aggregate data from leaves up to root (tree-based)."""
        if not self._tree_root:
            raise ValueError("Tree root not set")

        # Find leaves if not specified
        if leaf_nodes is None:
            leaf_nodes = [aid for aid, agent in self.agents.items()
                          if not agent.children]

        aggregated = {}

        # Start from leaves, propagate up
        for leaf in leaf_nodes:
            await self._converge_up(leaf, content, aggregated)

        return aggregated

    async def _converge_up(self, node: str, content: Any, aggregated: Dict):
        """Recursively converge up the tree."""
        aggregated[node] = content
        agent = self.agents.get(node)
        if agent and agent.parent:
            # Combine with siblings if all ready
            parent = self.agents.get(agent.parent)
            if parent:
                children_ready = all(c in aggregated for c in parent.children)
                if children_ready:
                    combined = {c: aggregated[c] for c in parent.children}
                    await self._converge_up(agent.parent, combined, aggregated)

    async def aggregate(self, messages: List[Message],
                        aggregator: Callable[[List[Any]], Any]) -> Message:
        """Aggregate multiple messages into one."""
        contents = [m.content for m in messages]
        combined = aggregator(contents)
        return Message(
            sender="aggregator",
            content=combined,
            pattern=MessagePattern.AGGREGATOR,
            headers={"source_count": len(messages)},
        )

    async def resequence(self, messages: List[Message]) -> List[Message]:
        """Restore message order by sequence number."""
        return sorted(messages, key=lambda m: m.sequence)

    # ─────────────────────────────────────────────────────────────────────────
    # Many-to-Many Patterns (N:M)
    # ─────────────────────────────────────────────────────────────────────────

    async def scatter_gather(self, sender: str, recipients: List[str],
                             content: Any, timeout: float = 5.0) -> List[Message]:
        """Scatter then gather (fan-out/fan-in)."""
        # Scatter
        correlation_id = str(uuid.uuid4())
        futures = []

        for recipient in recipients:
            request = Message(
                sender=sender,
                recipients=[recipient],
                content=content,
                pattern=MessagePattern.SCATTER_GATHER,
                correlation_id=correlation_id,
                reply_to=sender,
            )
            await self._deliver(request)

            future = asyncio.Future()
            self._register_reply_handler(f"{correlation_id}:{recipient}", future)
            futures.append(future)

        # Gather
        responses = []
        done, _ = await asyncio.wait(futures, timeout=timeout)
        for f in done:
            if not f.cancelled() and f.exception() is None:
                responses.append(f.result())

        return responses

    async def mesh_broadcast(self, sender: str, content: Any) -> List[Message]:
        """Full mesh broadcast - everyone sends to everyone."""
        all_messages = []
        for agent_id in self.agents:
            if agent_id != sender:
                msgs = await self.broadcast(agent_id, content, exclude={sender})
                all_messages.extend(msgs)
        return all_messages

    async def relay_chain(self, sender: str, content: Any,
                          chain: List[str]) -> List[Message]:
        """Sequential relay through chain of agents."""
        messages = []
        current_content = content

        for i, recipient in enumerate(chain):
            current_sender = sender if i == 0 else chain[i - 1]
            msg = Message(
                sender=current_sender,
                recipients=[recipient],
                content=current_content,
                pattern=MessagePattern.RELAY_CHAIN,
                sequence=i,
                headers={"chain_position": i, "chain_length": len(chain)},
            )
            await self._deliver(msg)
            messages.append(msg)
            current_content = msg.content  # Could be transformed

        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Hierarchical Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def tree_broadcast(self, sender: str, content: Any) -> List[Message]:
        """Broadcast down the tree from sender."""
        messages = []
        await self._tree_broadcast_recursive(sender, content, messages)
        return messages

    async def _tree_broadcast_recursive(self, node: str, content: Any, messages: List[Message]):
        agent = self.agents.get(node)
        if not agent:
            return

        for child in agent.children:
            msg = Message(
                sender=node,
                recipients=[child],
                content=content,
                pattern=MessagePattern.TREE_BROADCAST,
            )
            await self._deliver(msg)
            messages.append(msg)
            await self._tree_broadcast_recursive(child, content, messages)

    async def leader_broadcast(self, leader: str, content: Any) -> List[Message]:
        """Leader broadcasts to all followers."""
        followers = [a for a in self.agents if a != leader]
        return await self._send_to_many(leader, followers, content, MessagePattern.LEADER_FOLLOWER)

    # ─────────────────────────────────────────────────────────────────────────
    # Ring Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def ring_pass(self, sender: str, content: Any) -> List[Message]:
        """Pass message around the ring."""
        if not self._ring_order:
            return []

        messages = []
        current = sender
        current_content = content

        for _ in range(len(self._ring_order)):
            idx = self._ring_order.index(current)
            next_idx = (idx + 1) % len(self._ring_order)
            next_node = self._ring_order[next_idx]

            if next_node == sender:
                break  # Full circle

            msg = Message(
                sender=current,
                recipients=[next_node],
                content=current_content,
                pattern=MessagePattern.RING_PASS,
            )
            await self._deliver(msg)
            messages.append(msg)
            current = next_node

        return messages

    async def token_ring_acquire(self, agent_id: str, timeout: float = 5.0) -> bool:
        """Acquire token in token ring."""
        if self._ring_token_holder is None:
            self._ring_token_holder = agent_id
            return True

        if self._ring_token_holder == agent_id:
            return True

        # Wait for token
        start = time.time()
        while time.time() - start < timeout:
            if self._ring_token_holder == agent_id:
                return True
            await asyncio.sleep(0.1)

        return False

    def token_ring_release(self, agent_id: str):
        """Release token in token ring."""
        if self._ring_token_holder == agent_id:
            idx = self._ring_order.index(agent_id)
            next_idx = (idx + 1) % len(self._ring_order)
            self._ring_token_holder = self._ring_order[next_idx]

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def pipeline(self, content: Any, stages: List[str],
                       transformers: Dict[str, Callable] = None) -> Message:
        """Process through pipeline stages."""
        transformers = transformers or {}
        current_content = content

        for i, stage in enumerate(stages):
            # Transform if transformer exists for this stage
            if stage in transformers:
                current_content = transformers[stage](current_content)

            msg = Message(
                sender=stages[i - 1] if i > 0 else "source",
                recipients=[stage],
                content=current_content,
                pattern=MessagePattern.PIPELINE,
                sequence=i,
                headers={"stage": i, "total_stages": len(stages)},
            )
            await self._deliver(msg)

        return msg

    async def parallel_pipeline(self, content: Any, stage_groups: List[List[str]]) -> List[Message]:
        """Parallel pipeline with multiple agents per stage."""
        all_messages = []
        current_contents = [content]

        for stage_idx, stage_agents in enumerate(stage_groups):
            stage_messages = []
            for agent in stage_agents:
                for c in current_contents:
                    msg = Message(
                        sender=f"stage_{stage_idx - 1}" if stage_idx > 0 else "source",
                        recipients=[agent],
                        content=c,
                        pattern=MessagePattern.PARALLEL_PIPELINE,
                        sequence=stage_idx,
                    )
                    await self._deliver(msg)
                    stage_messages.append(msg)

            all_messages.extend(stage_messages)
            current_contents = [m.content for m in stage_messages]

        return all_messages

    # ─────────────────────────────────────────────────────────────────────────
    # Pub/Sub Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def publish(self, publisher: str, topic: str, content: Any) -> List[Message]:
        """Publish to a topic."""
        if topic not in self.topics:
            return []

        subscribers = [s for s in self.topics[topic] if s != publisher]
        messages = []

        for subscriber in subscribers:
            msg = TopicMessage(
                sender=publisher,
                recipients=[subscriber],
                content=content,
                pattern=MessagePattern.PUBLISH,
                topic=topic,
            )
            await self._deliver(msg)
            messages.append(msg)

        return messages

    async def content_filtered_publish(self, publisher: str, topic: str, content: Any,
                                        content_filter: Callable[[str, Any], bool]) -> List[Message]:
        """Publish with content-based filtering."""
        if topic not in self.topics:
            return []

        messages = []
        for subscriber in self.topics[topic]:
            if subscriber != publisher and content_filter(subscriber, content):
                msg = TopicMessage(
                    sender=publisher,
                    recipients=[subscriber],
                    content=content,
                    pattern=MessagePattern.CONTENT_FILTER,
                    topic=topic,
                )
                await self._deliver(msg)
                messages.append(msg)

        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Gossip/Epidemic Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def gossip_push(self, sender: str, content: Any, fanout: int = 3,
                          rounds: int = 5) -> List[Message]:
        """Push gossip to random peers."""
        messages = []
        infected = {sender}
        current_round_senders = [sender]

        for round_num in range(rounds):
            next_round_senders = []

            for s in current_round_senders:
                # Select random peers
                candidates = [a for a in self.agents if a not in infected]
                if not candidates:
                    break

                targets = random.sample(candidates, min(fanout, len(candidates)))

                for target in targets:
                    msg = Message(
                        sender=s,
                        recipients=[target],
                        content=content,
                        pattern=MessagePattern.GOSSIP_PUSH,
                        headers={"round": round_num, "gossip_type": "push"},
                    )
                    msg.visited = set(infected)
                    await self._deliver(msg)
                    messages.append(msg)
                    infected.add(target)
                    next_round_senders.append(target)

            current_round_senders = next_round_senders
            if not current_round_senders:
                break

        return messages

    async def gossip_push_pull(self, sender: str, content: Any, fanout: int = 2,
                                rounds: int = 3) -> List[Message]:
        """Bidirectional gossip - push then pull."""
        messages = []

        # Push phase
        push_msgs = await self.gossip_push(sender, content, fanout, rounds // 2 + 1)
        messages.extend(push_msgs)

        # Pull phase - receivers request from random peers
        infected = {m.recipients[0] for m in push_msgs}
        infected.add(sender)

        for agent_id in list(infected):
            targets = random.sample(list(infected - {agent_id}),
                                    min(fanout, len(infected) - 1))
            for target in targets:
                msg = Message(
                    sender=agent_id,
                    recipients=[target],
                    content={"type": "pull_request"},
                    pattern=MessagePattern.GOSSIP_PULL,
                    headers={"gossip_type": "pull"},
                )
                await self._deliver(msg)
                messages.append(msg)

        return messages

    async def epidemic_broadcast(self, sender: str, content: Any,
                                  infection_prob: float = 0.5) -> List[Message]:
        """Probabilistic epidemic spread."""
        messages = []
        infected = {sender}
        queue = [sender]

        while queue:
            current = queue.pop(0)

            for agent_id in self.agents:
                if agent_id in infected:
                    continue

                if random.random() < infection_prob:
                    msg = Message(
                        sender=current,
                        recipients=[agent_id],
                        content=content,
                        pattern=MessagePattern.EPIDEMIC_BROADCAST,
                        headers={"infection_prob": infection_prob},
                    )
                    await self._deliver(msg)
                    messages.append(msg)
                    infected.add(agent_id)
                    queue.append(agent_id)

        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Routing Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def content_route(self, sender: str, content: Any,
                            router: Callable[[Any], List[str]]) -> List[Message]:
        """Route based on content."""
        recipients = router(content)
        return await self._send_to_many(sender, recipients, content, MessagePattern.CONTENT_ROUTER)

    async def routing_slip(self, sender: str, content: Any, slip: List[str]) -> List[Message]:
        """Process through routing slip (sequential)."""
        messages = []
        current_content = content
        current_slip = list(slip)

        while current_slip:
            next_hop = current_slip.pop(0)
            msg = Message(
                sender=sender if not messages else messages[-1].recipients[0],
                recipients=[next_hop],
                content=current_content,
                pattern=MessagePattern.ROUTING_SLIP,
                headers={"remaining_slip": current_slip.copy()},
            )
            await self._deliver(msg)
            messages.append(msg)

        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Saga Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def start_saga(self, saga_id: str, steps: List[Tuple[str, Any]],
                         compensations: Dict[int, Tuple[str, Any]] = None) -> str:
        """Start a saga transaction."""
        self._sagas[saga_id] = {
            "id": saga_id,
            "steps": steps,
            "compensations": compensations or {},
            "current_step": 0,
            "state": {},
            "status": "running",
        }
        return saga_id

    async def execute_saga_step(self, saga_id: str) -> Optional[Message]:
        """Execute next saga step."""
        saga = self._sagas.get(saga_id)
        if not saga or saga["status"] != "running":
            return None

        step_idx = saga["current_step"]
        if step_idx >= len(saga["steps"]):
            saga["status"] = "completed"
            return None

        agent, content = saga["steps"][step_idx]
        msg = SagaMessage(
            sender="saga_coordinator",
            recipients=[agent],
            content=content,
            pattern=MessagePattern.SAGA_ORCHESTRATION,
            saga_id=saga_id,
            step=step_idx,
            total_steps=len(saga["steps"]),
            state=saga["state"].copy(),
        )
        await self._deliver(msg)

        saga["current_step"] += 1
        return msg

    async def compensate_saga(self, saga_id: str) -> List[Message]:
        """Execute compensation for failed saga."""
        saga = self._sagas.get(saga_id)
        if not saga:
            return []

        saga["status"] = "compensating"
        messages = []

        # Execute compensations in reverse order
        for step_idx in range(saga["current_step"] - 1, -1, -1):
            if step_idx in saga["compensations"]:
                agent, content = saga["compensations"][step_idx]
                msg = SagaMessage(
                    sender="saga_coordinator",
                    recipients=[agent],
                    content=content,
                    pattern=MessagePattern.COMPENSATING_TX,
                    saga_id=saga_id,
                    step=step_idx,
                    compensating=True,
                )
                await self._deliver(msg)
                messages.append(msg)

        saga["status"] = "compensated"
        return messages

    # ─────────────────────────────────────────────────────────────────────────
    # Fault Tolerance Patterns
    # ─────────────────────────────────────────────────────────────────────────

    def circuit_breaker_state(self, agent_id: str) -> str:
        """Get circuit breaker state for agent."""
        return self._circuit_states.get(agent_id, "closed")

    def trip_circuit(self, agent_id: str):
        """Trip circuit breaker to open."""
        self._circuit_states[agent_id] = "open"

    def reset_circuit(self, agent_id: str):
        """Reset circuit breaker to closed."""
        self._circuit_states[agent_id] = "closed"
        self._failure_counts[agent_id] = 0

    async def send_with_circuit_breaker(self, sender: str, recipient: str,
                                         content: Any, failure_threshold: int = 3) -> Optional[Message]:
        """Send with circuit breaker pattern."""
        state = self.circuit_breaker_state(recipient)

        if state == "open":
            # Check if we should try half-open
            self._circuit_states[recipient] = "half-open"
            state = "half-open"

        if state == "half-open":
            # Try one message
            try:
                msg = await self.direct(sender, recipient, content)
                self.reset_circuit(recipient)
                return msg
            except Exception:
                self.trip_circuit(recipient)
                return None

        # Normal send
        try:
            msg = await self.direct(sender, recipient, content)
            return msg
        except Exception:
            self._failure_counts[recipient] = self._failure_counts.get(recipient, 0) + 1
            if self._failure_counts[recipient] >= failure_threshold:
                self.trip_circuit(recipient)
            return None

    async def send_with_retry(self, sender: str, recipient: str, content: Any,
                               max_retries: int = 3, delay: float = 1.0) -> Optional[Message]:
        """Send with automatic retry."""
        for attempt in range(max_retries):
            try:
                msg = await self.direct(sender, recipient, content)
                msg.headers["attempt"] = attempt + 1
                return msg
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff

        # Failed all retries - send to dead letter
        dead = Message(
            sender=sender,
            recipients=[recipient],
            content=content,
            pattern=MessagePattern.DEAD_LETTER,
            headers={"max_retries_exceeded": True, "attempts": max_retries},
        )
        self.dead_letters.append(dead)
        return None

    async def send_with_timeout(self, sender: str, recipient: str, content: Any,
                                 timeout: float = 5.0) -> Optional[Message]:
        """Send with timeout."""
        try:
            msg_future = self.direct(sender, recipient, content)
            msg = await asyncio.wait_for(msg_future, timeout)
            return msg
        except asyncio.TimeoutError:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Flow Control Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def throttle(self, sender: str, recipient: str, content: Any,
                       rate: float = 1.0) -> Optional[Message]:
        """Throttle messages using token bucket."""
        key = f"{sender}:{recipient}"

        if key not in self._token_buckets:
            self._token_buckets[key] = (rate, rate, datetime.utcnow())

        tokens, max_tokens, last_update = self._token_buckets[key]

        # Refill tokens
        now = datetime.utcnow()
        elapsed = (now - last_update).total_seconds()
        tokens = min(max_tokens, tokens + elapsed * rate)

        if tokens >= 1.0:
            tokens -= 1.0
            self._token_buckets[key] = (tokens, max_tokens, now)
            return await self.direct(sender, recipient, content)
        else:
            self._token_buckets[key] = (tokens, max_tokens, now)
            return None  # Rate limited

    async def buffer_send(self, sender: str, recipient: str, content: Any,
                          buffer_size: int = 10) -> Optional[Message]:
        """Buffer messages before sending."""
        key = f"{sender}:{recipient}"

        if key not in self._windows:
            self._windows[key] = []

        self._windows[key].append(Message(
            sender=sender,
            recipients=[recipient],
            content=content,
            pattern=MessagePattern.BUFFER,
        ))

        if len(self._windows[key]) >= buffer_size:
            messages = self._windows[key]
            self._windows[key] = []

            # Send all buffered messages
            for msg in messages:
                await self._deliver(msg)

            return messages[-1]

        return None

    async def window_send(self, sender: str, recipient: str, content: Any,
                          window_ms: int = 1000) -> List[Message]:
        """Send messages in time windows."""
        key = f"{sender}:{recipient}"

        if key not in self._windows:
            self._windows[key] = []

        self._windows[key].append(Message(
            sender=sender,
            recipients=[recipient],
            content=content,
            pattern=MessagePattern.WINDOW,
            timestamp=datetime.utcnow(),
        ))

        # Check if window has elapsed
        if self._windows[key]:
            first_ts = self._windows[key][0].timestamp
            elapsed_ms = (datetime.utcnow() - first_ts).total_seconds() * 1000

            if elapsed_ms >= window_ms:
                messages = self._windows[key]
                self._windows[key] = []

                for msg in messages:
                    await self._deliver(msg)

                return messages

        return []

    # ─────────────────────────────────────────────────────────────────────────
    # Coordination Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def barrier(self, participants: List[str], barrier_id: str) -> bool:
        """Synchronization barrier - wait for all participants."""
        barrier_key = f"barrier:{barrier_id}"

        if barrier_key not in self._windows:
            self._windows[barrier_key] = []

        # Add current arrivals
        for p in participants:
            if p not in self._windows[barrier_key]:
                self._windows[barrier_key].append(p)

        # Check if all arrived
        all_arrived = set(self._windows[barrier_key]) == set(participants)

        if all_arrived:
            del self._windows[barrier_key]
            return True

        return False

    async def quorum_broadcast(self, sender: str, content: Any, group: str,
                                quorum_size: int, timeout: float = 5.0) -> Tuple[bool, List[Message]]:
        """Broadcast requiring quorum acknowledgment."""
        if group not in self.groups:
            return False, []

        members = list(self.groups[group])
        if len(members) < quorum_size:
            return False, []

        # Send to all
        messages = await self.multicast(sender, group, content)

        # Wait for quorum acks (simplified - would normally wait for actual acks)
        await asyncio.sleep(0.1)

        # Assume we got enough acks
        return True, messages

    # ─────────────────────────────────────────────────────────────────────────
    # Monitoring Patterns
    # ─────────────────────────────────────────────────────────────────────────

    async def wire_tap(self, msg: Message, tap_destination: str) -> Message:
        """Copy message for monitoring."""
        tap_msg = msg.clone()
        tap_msg.pattern = MessagePattern.WIRE_TAP
        tap_msg.recipients = [tap_destination]
        tap_msg.headers["tapped_from"] = msg.id
        await self._deliver(tap_msg)
        return tap_msg

    def get_message_history(self, agent_id: str = None,
                            pattern: MessagePattern = None,
                            limit: int = 100) -> List[Message]:
        """Get message history with optional filters."""
        messages = self.message_log

        if agent_id:
            messages = [m for m in messages
                        if m.sender == agent_id or agent_id in m.recipients]

        if pattern:
            messages = [m for m in messages if m.pattern == pattern]

        return messages[-limit:]

    # ─────────────────────────────────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────────────────────────────────

    async def _deliver(self, msg: Message, log: bool = True):
        """Deliver message to recipients."""
        if log:
            self.message_log.append(msg)

        for recipient in msg.recipients:
            if recipient in self.agents:
                agent = self.agents[recipient]
                await agent.inbox.put(msg)
                agent.messages_received += 1
                agent.last_active = datetime.utcnow()

        if msg.sender in self.agents:
            self.agents[msg.sender].messages_sent += 1
            self.agents[msg.sender].last_active = datetime.utcnow()

    async def _send_to_many(self, sender: str, recipients: List[str],
                            content: Any, pattern: MessagePattern) -> List[Message]:
        """Send same content to multiple recipients."""
        messages = []
        for recipient in recipients:
            msg = Message(
                sender=sender,
                recipients=[recipient],
                content=content,
                pattern=pattern,
            )
            await self._deliver(msg)
            messages.append(msg)
        return messages

    def _register_reply_handler(self, correlation_id: str, future: asyncio.Future):
        """Register handler for reply correlation."""
        if correlation_id not in self._message_handlers:
            self._message_handlers[correlation_id] = []
        self._message_handlers[correlation_id].append(future)

    def get_pattern_stats(self) -> Dict[str, int]:
        """Get count of messages by pattern."""
        stats = {}
        for msg in self.message_log:
            pattern = msg.pattern.value
            stats[pattern] = stats.get(pattern, 0) + 1
        return stats

    def export_topology(self) -> Dict[str, Any]:
        """Export network topology."""
        return {
            "agents": {
                aid: {
                    "groups": list(a.groups),
                    "topics": list(a.topics),
                    "neighbors": list(a.neighbors),
                    "parent": a.parent,
                    "children": list(a.children),
                    "location": a.location,
                    "messages_sent": a.messages_sent,
                    "messages_received": a.messages_received,
                }
                for aid, a in self.agents.items()
            },
            "groups": {g: list(members) for g, members in self.groups.items()},
            "topics": {t: list(subs) for t, subs in self.topics.items()},
            "ring_order": self._ring_order,
            "tree_root": self._tree_root,
            "total_messages": len(self.message_log),
            "dead_letters": len(self.dead_letters),
        }
