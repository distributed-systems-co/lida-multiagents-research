"""Locality-based messaging and quasi-quorum formation.

Features:
- Mailbox system for asynchronous agent communication
- Location/proximity-based message routing
- Local message exchange without global logging
- Quasi-quorum formation for distributed consensus
- Neighborhood topology management
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import random
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import heapq


class MessagePriority(int, Enum):
    """Message priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


class DeliveryMode(str, Enum):
    """Message delivery modes."""
    DIRECT = "direct"           # Point-to-point
    BROADCAST = "broadcast"     # To all agents
    MULTICAST = "multicast"     # To a group
    LOCAL = "local"             # To nearby agents only
    GOSSIP = "gossip"           # Probabilistic propagation
    QUORUM = "quorum"           # To quorum members


class QuorumState(str, Enum):
    """Quorum formation states."""
    FORMING = "forming"
    ACTIVE = "active"
    VOTING = "voting"
    DECIDED = "decided"
    DISSOLVED = "dissolved"


@dataclass
class Location:
    """N-dimensional location in agent space."""

    coordinates: Tuple[float, ...] = (0.0, 0.0)
    space_id: str = "default"

    def distance_to(self, other: "Location") -> float:
        """Euclidean distance to another location."""
        if len(self.coordinates) != len(other.coordinates):
            return float('inf')
        return math.sqrt(sum(
            (a - b) ** 2 for a, b in zip(self.coordinates, other.coordinates)
        ))

    def within_radius(self, other: "Location", radius: float) -> bool:
        """Check if within radius of another location."""
        return self.distance_to(other) <= radius

    def move_towards(self, target: "Location", step: float) -> "Location":
        """Move towards target location."""
        dist = self.distance_to(target)
        if dist == 0:
            return self

        ratio = min(step / dist, 1.0)
        new_coords = tuple(
            a + ratio * (b - a)
            for a, b in zip(self.coordinates, target.coordinates)
        )
        return Location(coordinates=new_coords, space_id=self.space_id)


@dataclass
class Message:
    """A message in the mailbox system."""

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast/multicast

    # Content
    content: Any = None
    content_type: str = "generic"

    # Delivery
    delivery_mode: DeliveryMode = DeliveryMode.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    ttl: int = 100  # Time to live in ticks

    # Groups/targeting
    target_group: Optional[str] = None
    target_radius: Optional[float] = None  # For local delivery

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    tick: int = 0
    hops: int = 0
    max_hops: int = 10
    visited: Set[str] = field(default_factory=set)

    # Flags
    logged: bool = True  # Whether to log this message
    acknowledged: bool = False
    expires_at: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """Comparison for priority queue."""
        return self.priority.value > other.priority.value


@dataclass
class Mailbox:
    """Agent mailbox for receiving messages."""

    owner_id: str
    capacity: int = 1000

    # Message queues by priority
    inbox: List[Message] = field(default_factory=list)
    outbox: List[Message] = field(default_factory=list)

    # Processed messages
    processed: List[str] = field(default_factory=list)
    max_processed_history: int = 100

    # Statistics
    received_count: int = 0
    sent_count: int = 0
    dropped_count: int = 0

    def receive(self, message: Message) -> bool:
        """Receive a message into inbox."""
        if len(self.inbox) >= self.capacity:
            # Drop lowest priority message
            if self.inbox and message.priority > min(m.priority for m in self.inbox):
                min_msg = min(self.inbox, key=lambda m: m.priority)
                self.inbox.remove(min_msg)
                self.dropped_count += 1
            else:
                self.dropped_count += 1
                return False

        heapq.heappush(self.inbox, message)
        self.received_count += 1
        return True

    def pop(self) -> Optional[Message]:
        """Get highest priority message."""
        if not self.inbox:
            return None

        message = heapq.heappop(self.inbox)
        self.processed.append(message.message_id)

        # Trim history
        if len(self.processed) > self.max_processed_history:
            self.processed = self.processed[-self.max_processed_history:]

        return message

    def peek(self, n: int = 1) -> List[Message]:
        """Peek at top n messages without removing."""
        return sorted(self.inbox, key=lambda m: -m.priority.value)[:n]

    def has_message(self, message_id: str) -> bool:
        """Check if message was already processed."""
        return message_id in self.processed or any(
            m.message_id == message_id for m in self.inbox
        )

    def queue_send(self, message: Message):
        """Queue a message for sending."""
        self.outbox.append(message)
        self.sent_count += 1

    def flush_outbox(self) -> List[Message]:
        """Get and clear outbox."""
        messages = self.outbox
        self.outbox = []
        return messages


@dataclass
class AgentNode:
    """An agent node in the locality network."""

    agent_id: str
    location: Location = field(default_factory=Location)
    mailbox: Mailbox = field(default=None)

    # Neighborhood
    neighbors: Set[str] = field(default_factory=set)
    neighbor_distances: Dict[str, float] = field(default_factory=dict)

    # Group memberships
    groups: Set[str] = field(default_factory=set)
    quorums: Set[str] = field(default_factory=set)

    # Communication settings
    communication_radius: float = 10.0
    local_only_radius: float = 3.0  # Messages in this radius not logged

    # Status
    active: bool = True
    last_seen: datetime = field(default_factory=datetime.now)

    # Metadata
    capabilities: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mailbox is None:
            self.mailbox = Mailbox(owner_id=self.agent_id)


@dataclass
class QuorumVote:
    """A vote in a quorum decision."""

    voter_id: str
    proposal_id: str
    vote: bool  # True = approve, False = reject
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    signature: str = ""  # For verification


@dataclass
class QuorumProposal:
    """A proposal for quorum voting."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    proposer_id: str = ""
    content: Any = None
    proposal_type: str = "generic"

    # Voting
    votes: Dict[str, QuorumVote] = field(default_factory=dict)
    threshold: float = 0.5  # Fraction needed to pass
    min_voters: int = 3

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    decided_at: Optional[datetime] = None

    # Result
    passed: Optional[bool] = None
    final_tally: Dict[str, int] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Quorum:
    """A quasi-quorum for distributed decisions."""

    quorum_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""

    # Members
    members: Set[str] = field(default_factory=set)
    member_weights: Dict[str, float] = field(default_factory=dict)
    min_size: int = 3
    max_size: int = 100

    # State
    state: QuorumState = QuorumState.FORMING
    leader_id: Optional[str] = None

    # Proposals
    active_proposals: Dict[str, QuorumProposal] = field(default_factory=dict)
    completed_proposals: List[str] = field(default_factory=list)

    # Configuration
    consensus_threshold: float = 0.67  # 2/3 majority
    vote_timeout_seconds: int = 60
    require_leader: bool = False

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_weight(self) -> float:
        """Get total voting weight."""
        return sum(self.member_weights.get(m, 1.0) for m in self.members)

    def add_member(self, agent_id: str, weight: float = 1.0) -> bool:
        """Add a member to the quorum."""
        if len(self.members) >= self.max_size:
            return False

        self.members.add(agent_id)
        self.member_weights[agent_id] = weight

        if len(self.members) >= self.min_size and self.state == QuorumState.FORMING:
            self.state = QuorumState.ACTIVE

        return True

    def remove_member(self, agent_id: str):
        """Remove a member from the quorum."""
        self.members.discard(agent_id)
        self.member_weights.pop(agent_id, None)

        if len(self.members) < self.min_size:
            self.state = QuorumState.FORMING

    def create_proposal(
        self,
        proposer_id: str,
        content: Any,
        proposal_type: str = "generic",
        deadline_seconds: int = None,
    ) -> Optional[QuorumProposal]:
        """Create a new proposal for voting."""
        if proposer_id not in self.members:
            return None

        if self.state != QuorumState.ACTIVE:
            return None

        proposal = QuorumProposal(
            proposer_id=proposer_id,
            content=content,
            proposal_type=proposal_type,
            threshold=self.consensus_threshold,
            min_voters=max(1, int(len(self.members) * 0.5)),
        )

        if deadline_seconds:
            proposal.deadline = datetime.now() + timedelta(seconds=deadline_seconds)
        else:
            proposal.deadline = datetime.now() + timedelta(seconds=self.vote_timeout_seconds)

        self.active_proposals[proposal.proposal_id] = proposal
        self.state = QuorumState.VOTING

        return proposal

    def cast_vote(
        self,
        voter_id: str,
        proposal_id: str,
        vote: bool,
    ) -> Optional[QuorumVote]:
        """Cast a vote on a proposal."""
        if voter_id not in self.members:
            return None

        if proposal_id not in self.active_proposals:
            return None

        proposal = self.active_proposals[proposal_id]

        # Check deadline
        if proposal.deadline and datetime.now() > proposal.deadline:
            self._finalize_proposal(proposal_id)
            return None

        vote_obj = QuorumVote(
            voter_id=voter_id,
            proposal_id=proposal_id,
            vote=vote,
            weight=self.member_weights.get(voter_id, 1.0),
        )

        proposal.votes[voter_id] = vote_obj

        # Check if we can decide early
        self._check_early_decision(proposal_id)

        return vote_obj

    def _check_early_decision(self, proposal_id: str):
        """Check if proposal can be decided before deadline."""
        proposal = self.active_proposals.get(proposal_id)
        if not proposal or proposal.passed is not None:
            return

        total = self.total_weight()
        yes_weight = sum(
            v.weight for v in proposal.votes.values() if v.vote
        )
        no_weight = sum(
            v.weight for v in proposal.votes.values() if not v.vote
        )

        # Can we reach threshold with remaining votes?
        voted_weight = yes_weight + no_weight
        remaining = total - voted_weight

        # Definite pass
        if yes_weight / total >= proposal.threshold:
            self._finalize_proposal(proposal_id, passed=True)
        # Definite fail
        elif (yes_weight + remaining) / total < proposal.threshold:
            self._finalize_proposal(proposal_id, passed=False)
        # All votes in
        elif voted_weight >= total * proposal.min_voters / len(self.members):
            if len(proposal.votes) >= proposal.min_voters:
                passed = yes_weight / voted_weight >= proposal.threshold
                self._finalize_proposal(proposal_id, passed=passed)

    def _finalize_proposal(self, proposal_id: str, passed: Optional[bool] = None):
        """Finalize a proposal."""
        proposal = self.active_proposals.get(proposal_id)
        if not proposal:
            return

        if passed is None:
            # Calculate from votes
            total_voted = sum(v.weight for v in proposal.votes.values())
            yes_weight = sum(v.weight for v in proposal.votes.values() if v.vote)
            passed = (yes_weight / total_voted >= proposal.threshold) if total_voted > 0 else False

        proposal.passed = passed
        proposal.decided_at = datetime.now()
        proposal.final_tally = {
            "yes": sum(1 for v in proposal.votes.values() if v.vote),
            "no": sum(1 for v in proposal.votes.values() if not v.vote),
            "yes_weight": sum(v.weight for v in proposal.votes.values() if v.vote),
            "no_weight": sum(v.weight for v in proposal.votes.values() if not v.vote),
        }

        self.completed_proposals.append(proposal_id)
        del self.active_proposals[proposal_id]

        if not self.active_proposals:
            self.state = QuorumState.ACTIVE


class LocalityNetwork:
    """Network managing agent locations, mailboxes, and quorums."""

    def __init__(self, default_radius: float = 10.0):
        self.default_radius = default_radius

        # Agents
        self._agents: Dict[str, AgentNode] = {}

        # Groups
        self._groups: Dict[str, Set[str]] = defaultdict(set)

        # Quorums
        self._quorums: Dict[str, Quorum] = {}

        # Message log (only for logged messages)
        self._message_log: List[Message] = []
        self._max_log_size: int = 10000

        # Local message count (not logged but tracked)
        self._local_message_count: int = 0

        # Spatial index for fast neighbor lookup
        self._spatial_index: Dict[str, Dict[Tuple[int, ...], Set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._cell_size: float = 5.0

        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "local_exchanges": 0,
            "broadcasts": 0,
            "quorum_decisions": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Management
    # ─────────────────────────────────────────────────────────────────────────

    def register_agent(
        self,
        agent_id: str,
        location: Optional[Location] = None,
        **kwargs,
    ) -> AgentNode:
        """Register an agent in the network."""
        if agent_id in self._agents:
            return self._agents[agent_id]

        node = AgentNode(
            agent_id=agent_id,
            location=location or Location(),
            **kwargs,
        )

        self._agents[agent_id] = node
        self._update_spatial_index(agent_id, None, node.location)
        self._update_neighbors(agent_id)

        return node

    def unregister_agent(self, agent_id: str):
        """Remove an agent from the network."""
        if agent_id not in self._agents:
            return

        node = self._agents[agent_id]

        # Remove from spatial index
        cell = self._get_cell(node.location)
        self._spatial_index[node.location.space_id][cell].discard(agent_id)

        # Remove from groups and quorums
        for group_id in list(node.groups):
            self._groups[group_id].discard(agent_id)

        for quorum_id in list(node.quorums):
            if quorum_id in self._quorums:
                self._quorums[quorum_id].remove_member(agent_id)

        # Remove from neighbors' lists
        for neighbor_id in node.neighbors:
            if neighbor_id in self._agents:
                self._agents[neighbor_id].neighbors.discard(agent_id)
                self._agents[neighbor_id].neighbor_distances.pop(agent_id, None)

        del self._agents[agent_id]

    def move_agent(self, agent_id: str, new_location: Location):
        """Move an agent to a new location."""
        if agent_id not in self._agents:
            return

        node = self._agents[agent_id]
        old_location = node.location

        self._update_spatial_index(agent_id, old_location, new_location)
        node.location = new_location
        self._update_neighbors(agent_id)

    def _get_cell(self, location: Location) -> Tuple[int, ...]:
        """Get spatial index cell for a location."""
        return tuple(int(c / self._cell_size) for c in location.coordinates)

    def _update_spatial_index(
        self,
        agent_id: str,
        old_location: Optional[Location],
        new_location: Location,
    ):
        """Update spatial index for agent movement."""
        if old_location:
            old_cell = self._get_cell(old_location)
            self._spatial_index[old_location.space_id][old_cell].discard(agent_id)

        new_cell = self._get_cell(new_location)
        self._spatial_index[new_location.space_id][new_cell].add(agent_id)

    def _update_neighbors(self, agent_id: str):
        """Update neighbor relationships for an agent."""
        if agent_id not in self._agents:
            return

        node = self._agents[agent_id]
        old_neighbors = node.neighbors.copy()
        node.neighbors.clear()
        node.neighbor_distances.clear()

        # Check nearby cells
        cell = self._get_cell(node.location)
        cells_to_check = self._get_nearby_cells(cell)

        for check_cell in cells_to_check:
            for other_id in self._spatial_index[node.location.space_id][check_cell]:
                if other_id == agent_id:
                    continue

                other = self._agents.get(other_id)
                if not other:
                    continue

                dist = node.location.distance_to(other.location)
                if dist <= node.communication_radius:
                    node.neighbors.add(other_id)
                    node.neighbor_distances[other_id] = dist

                    # Bidirectional
                    if dist <= other.communication_radius:
                        other.neighbors.add(agent_id)
                        other.neighbor_distances[agent_id] = dist

        # Clean up old neighbors
        for old_id in old_neighbors - node.neighbors:
            if old_id in self._agents:
                self._agents[old_id].neighbors.discard(agent_id)
                self._agents[old_id].neighbor_distances.pop(agent_id, None)

    def _get_nearby_cells(self, cell: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Get cells within communication range."""
        # For 2D, check 3x3 grid around cell
        offsets = [-1, 0, 1]
        cells = []

        from itertools import product
        for offset in product(*[offsets] * len(cell)):
            nearby = tuple(c + o for c, o in zip(cell, offset))
            cells.append(nearby)

        return cells

    def get_neighbors(self, agent_id: str, max_distance: Optional[float] = None) -> List[str]:
        """Get neighbors of an agent."""
        if agent_id not in self._agents:
            return []

        node = self._agents[agent_id]

        if max_distance is None:
            return list(node.neighbors)

        return [
            nid for nid, dist in node.neighbor_distances.items()
            if dist <= max_distance
        ]

    def get_local_neighbors(self, agent_id: str) -> List[str]:
        """Get very close neighbors (for unlogged communication)."""
        if agent_id not in self._agents:
            return []

        node = self._agents[agent_id]
        return [
            nid for nid, dist in node.neighbor_distances.items()
            if dist <= node.local_only_radius
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Messaging
    # ─────────────────────────────────────────────────────────────────────────

    def send_message(
        self,
        sender_id: str,
        recipient_id: Optional[str],
        content: Any,
        delivery_mode: DeliveryMode = DeliveryMode.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        target_group: Optional[str] = None,
        target_radius: Optional[float] = None,
        logged: bool = True,
        **metadata,
    ) -> Optional[Message]:
        """Send a message through the network."""
        if sender_id not in self._agents:
            return None

        sender = self._agents[sender_id]

        message = Message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            delivery_mode=delivery_mode,
            priority=priority,
            target_group=target_group,
            target_radius=target_radius,
            logged=logged,
            metadata=metadata,
        )

        # Determine if this should be logged
        if delivery_mode == DeliveryMode.LOCAL:
            message.logged = False
        elif delivery_mode == DeliveryMode.DIRECT and recipient_id:
            # Check if recipient is very close
            if recipient_id in sender.neighbor_distances:
                if sender.neighbor_distances[recipient_id] <= sender.local_only_radius:
                    message.logged = False

        self._stats["messages_sent"] += 1

        # Route based on delivery mode
        if delivery_mode == DeliveryMode.DIRECT:
            self._deliver_direct(message)
        elif delivery_mode == DeliveryMode.BROADCAST:
            self._deliver_broadcast(message)
        elif delivery_mode == DeliveryMode.MULTICAST:
            self._deliver_multicast(message)
        elif delivery_mode == DeliveryMode.LOCAL:
            self._deliver_local(message)
        elif delivery_mode == DeliveryMode.GOSSIP:
            self._deliver_gossip(message)
        elif delivery_mode == DeliveryMode.QUORUM:
            self._deliver_quorum(message)

        # Log if needed
        if message.logged:
            self._log_message(message)
        else:
            self._local_message_count += 1
            self._stats["local_exchanges"] += 1

        return message

    def _deliver_direct(self, message: Message):
        """Deliver message directly to recipient."""
        if not message.recipient_id:
            return

        if message.recipient_id in self._agents:
            recipient = self._agents[message.recipient_id]
            if recipient.mailbox.receive(message):
                self._stats["messages_delivered"] += 1

    def _deliver_broadcast(self, message: Message):
        """Broadcast message to all agents."""
        self._stats["broadcasts"] += 1

        for agent_id, agent in self._agents.items():
            if agent_id == message.sender_id:
                continue

            msg_copy = Message(
                message_id=message.message_id,
                sender_id=message.sender_id,
                recipient_id=agent_id,
                content=message.content,
                content_type=message.content_type,
                delivery_mode=message.delivery_mode,
                priority=message.priority,
                logged=message.logged,
                metadata=message.metadata.copy(),
            )

            if agent.mailbox.receive(msg_copy):
                self._stats["messages_delivered"] += 1

    def _deliver_multicast(self, message: Message):
        """Deliver message to a group."""
        if not message.target_group:
            return

        group_members = self._groups.get(message.target_group, set())

        for member_id in group_members:
            if member_id == message.sender_id:
                continue

            if member_id in self._agents:
                msg_copy = Message(
                    message_id=message.message_id,
                    sender_id=message.sender_id,
                    recipient_id=member_id,
                    content=message.content,
                    content_type=message.content_type,
                    delivery_mode=message.delivery_mode,
                    priority=message.priority,
                    target_group=message.target_group,
                    logged=message.logged,
                    metadata=message.metadata.copy(),
                )

                if self._agents[member_id].mailbox.receive(msg_copy):
                    self._stats["messages_delivered"] += 1

    def _deliver_local(self, message: Message):
        """Deliver message only to nearby agents."""
        sender = self._agents.get(message.sender_id)
        if not sender:
            return

        radius = message.target_radius or sender.local_only_radius

        for neighbor_id, dist in sender.neighbor_distances.items():
            if dist > radius:
                continue

            if neighbor_id in self._agents:
                msg_copy = Message(
                    message_id=message.message_id,
                    sender_id=message.sender_id,
                    recipient_id=neighbor_id,
                    content=message.content,
                    content_type=message.content_type,
                    delivery_mode=message.delivery_mode,
                    priority=message.priority,
                    target_radius=radius,
                    logged=False,  # Local messages never logged
                    metadata=message.metadata.copy(),
                )

                if self._agents[neighbor_id].mailbox.receive(msg_copy):
                    self._stats["messages_delivered"] += 1

    def _deliver_gossip(self, message: Message):
        """Probabilistic gossip propagation."""
        sender = self._agents.get(message.sender_id)
        if not sender:
            return

        if message.hops >= message.max_hops:
            return

        # Select random subset of neighbors
        neighbors = list(sender.neighbors - message.visited)
        fanout = min(3, len(neighbors))  # Send to up to 3 neighbors

        if fanout == 0:
            return

        selected = random.sample(neighbors, fanout)

        for neighbor_id in selected:
            msg_copy = Message(
                message_id=message.message_id,
                sender_id=message.sender_id,
                recipient_id=neighbor_id,
                content=message.content,
                content_type=message.content_type,
                delivery_mode=message.delivery_mode,
                priority=message.priority,
                ttl=message.ttl - 1,
                hops=message.hops + 1,
                max_hops=message.max_hops,
                visited=message.visited | {message.sender_id},
                logged=message.logged,
                metadata=message.metadata.copy(),
            )

            if neighbor_id in self._agents:
                neighbor = self._agents[neighbor_id]

                # Don't re-deliver
                if neighbor.mailbox.has_message(message.message_id):
                    continue

                if neighbor.mailbox.receive(msg_copy):
                    self._stats["messages_delivered"] += 1

                    # Continue gossip from this neighbor
                    msg_copy.sender_id = neighbor_id
                    self._deliver_gossip(msg_copy)

    def _deliver_quorum(self, message: Message):
        """Deliver message to quorum members."""
        if not message.target_group:
            return

        quorum = self._quorums.get(message.target_group)
        if not quorum:
            return

        for member_id in quorum.members:
            if member_id == message.sender_id:
                continue

            if member_id in self._agents:
                msg_copy = Message(
                    message_id=message.message_id,
                    sender_id=message.sender_id,
                    recipient_id=member_id,
                    content=message.content,
                    content_type=message.content_type,
                    delivery_mode=message.delivery_mode,
                    priority=message.priority,
                    target_group=message.target_group,
                    logged=message.logged,
                    metadata=message.metadata.copy(),
                )

                if self._agents[member_id].mailbox.receive(msg_copy):
                    self._stats["messages_delivered"] += 1

    def _log_message(self, message: Message):
        """Log a message to the message log."""
        self._message_log.append(message)

        # Trim log
        if len(self._message_log) > self._max_log_size:
            self._message_log = self._message_log[-self._max_log_size:]

    def get_messages(
        self,
        agent_id: str,
        limit: int = 10,
    ) -> List[Message]:
        """Get messages from agent's mailbox."""
        if agent_id not in self._agents:
            return []

        agent = self._agents[agent_id]
        return agent.mailbox.peek(limit)

    def pop_message(self, agent_id: str) -> Optional[Message]:
        """Pop highest priority message from agent's mailbox."""
        if agent_id not in self._agents:
            return None

        return self._agents[agent_id].mailbox.pop()

    # ─────────────────────────────────────────────────────────────────────────
    # Groups
    # ─────────────────────────────────────────────────────────────────────────

    def create_group(self, group_id: str, member_ids: List[str] = None) -> str:
        """Create a group."""
        for member_id in (member_ids or []):
            if member_id in self._agents:
                self._groups[group_id].add(member_id)
                self._agents[member_id].groups.add(group_id)

        return group_id

    def join_group(self, agent_id: str, group_id: str):
        """Add agent to a group."""
        if agent_id in self._agents:
            self._groups[group_id].add(agent_id)
            self._agents[agent_id].groups.add(group_id)

    def leave_group(self, agent_id: str, group_id: str):
        """Remove agent from a group."""
        self._groups[group_id].discard(agent_id)
        if agent_id in self._agents:
            self._agents[agent_id].groups.discard(group_id)

    def get_group_members(self, group_id: str) -> List[str]:
        """Get members of a group."""
        return list(self._groups.get(group_id, set()))

    # ─────────────────────────────────────────────────────────────────────────
    # Quorums
    # ─────────────────────────────────────────────────────────────────────────

    def create_quorum(
        self,
        quorum_id: Optional[str] = None,
        name: str = "",
        members: List[str] = None,
        consensus_threshold: float = 0.67,
        min_size: int = 3,
    ) -> Quorum:
        """Create a quasi-quorum."""
        quorum = Quorum(
            quorum_id=quorum_id or str(uuid.uuid4())[:12],
            name=name or f"quorum_{len(self._quorums)}",
            consensus_threshold=consensus_threshold,
            min_size=min_size,
        )

        for member_id in (members or []):
            if member_id in self._agents:
                quorum.add_member(member_id)
                self._agents[member_id].quorums.add(quorum.quorum_id)

        self._quorums[quorum.quorum_id] = quorum
        return quorum

    def join_quorum(self, agent_id: str, quorum_id: str, weight: float = 1.0) -> bool:
        """Add agent to a quorum."""
        if quorum_id not in self._quorums:
            return False

        if agent_id not in self._agents:
            return False

        quorum = self._quorums[quorum_id]
        if quorum.add_member(agent_id, weight):
            self._agents[agent_id].quorums.add(quorum_id)
            return True

        return False

    def leave_quorum(self, agent_id: str, quorum_id: str):
        """Remove agent from a quorum."""
        if quorum_id in self._quorums:
            self._quorums[quorum_id].remove_member(agent_id)

        if agent_id in self._agents:
            self._agents[agent_id].quorums.discard(quorum_id)

    def propose(
        self,
        agent_id: str,
        quorum_id: str,
        content: Any,
        proposal_type: str = "generic",
        deadline_seconds: int = None,
    ) -> Optional[QuorumProposal]:
        """Create a proposal in a quorum."""
        if quorum_id not in self._quorums:
            return None

        quorum = self._quorums[quorum_id]
        return quorum.create_proposal(
            agent_id,
            content,
            proposal_type,
            deadline_seconds,
        )

    def vote(
        self,
        agent_id: str,
        quorum_id: str,
        proposal_id: str,
        vote: bool,
    ) -> Optional[QuorumVote]:
        """Cast a vote on a proposal."""
        if quorum_id not in self._quorums:
            return None

        quorum = self._quorums[quorum_id]
        vote_obj = quorum.cast_vote(agent_id, proposal_id, vote)

        # Check if decision was made
        if proposal_id in quorum.completed_proposals:
            self._stats["quorum_decisions"] += 1

        return vote_obj

    def get_quorum(self, quorum_id: str) -> Optional[Quorum]:
        """Get a quorum by ID."""
        return self._quorums.get(quorum_id)

    def get_active_proposals(self, quorum_id: str) -> List[QuorumProposal]:
        """Get active proposals in a quorum."""
        quorum = self._quorums.get(quorum_id)
        if not quorum:
            return []

        return list(quorum.active_proposals.values())

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            **self._stats,
            "agents": len(self._agents),
            "groups": len(self._groups),
            "quorums": len(self._quorums),
            "logged_messages": len(self._message_log),
            "local_messages": self._local_message_count,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export network state to dictionary."""
        return {
            "agents": [
                {
                    "id": a.agent_id,
                    "location": list(a.location.coordinates),
                    "neighbors": list(a.neighbors),
                    "groups": list(a.groups),
                    "quorums": list(a.quorums),
                    "mailbox_size": len(a.mailbox.inbox),
                }
                for a in self._agents.values()
            ],
            "groups": {
                gid: list(members)
                for gid, members in self._groups.items()
            },
            "quorums": [
                {
                    "id": q.quorum_id,
                    "name": q.name,
                    "state": q.state.value,
                    "members": list(q.members),
                    "active_proposals": len(q.active_proposals),
                    "completed_proposals": len(q.completed_proposals),
                }
                for q in self._quorums.values()
            ],
            "stats": self._stats,
        }


# Global locality network instance
_locality_network: Optional[LocalityNetwork] = None


def get_locality_network() -> LocalityNetwork:
    """Get or create the global locality network."""
    global _locality_network
    if _locality_network is None:
        _locality_network = LocalityNetwork()
    return _locality_network


def reset_locality_network() -> LocalityNetwork:
    """Reset the global locality network."""
    global _locality_network
    _locality_network = LocalityNetwork()
    return _locality_network
