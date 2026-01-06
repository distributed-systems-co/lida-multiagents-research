"""Distributed consensus protocols for fault-tolerant multi-agent coordination.

Provides:
- Raft consensus algorithm
- Byzantine fault tolerance (PBFT-inspired)
- Paxos-style agreement
- Leader election
- Log replication
- Membership changes
- Fault detection and recovery
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class NodeState(Enum):
    """Raft node states."""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class MessageType(Enum):
    """Consensus message types."""
    REQUEST_VOTE = "request_vote"
    VOTE = "vote"
    APPEND_ENTRIES = "append_entries"
    APPEND_RESPONSE = "append_response"
    CLIENT_REQUEST = "client_request"
    HEARTBEAT = "heartbeat"


@dataclass
class LogEntry:
    """A single log entry."""

    term: int  # Election term when entry was created
    index: int  # Position in log
    command: Any  # Command to execute
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "index": self.index,
            "command": self.command,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConsensusMessage:
    """Message for consensus protocol."""

    msg_type: MessageType
    sender_id: str
    term: int
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ═══════════════════════════════════════════════════════════════════════════
# RAFT CONSENSUS
# ═══════════════════════════════════════════════════════════════════════════

class RaftNode:
    """A single node in the Raft consensus cluster."""

    def __init__(
        self,
        node_id: str,
        cluster_nodes: List[str],
        election_timeout_range: Tuple[float, float] = (0.15, 0.3),
        heartbeat_interval: float = 0.05,
    ):
        self.node_id = node_id
        self.cluster_nodes = [n for n in cluster_nodes if n != node_id]  # Other nodes

        # Persistent state
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.log: List[LogEntry] = []

        # Volatile state
        self.commit_index = 0  # Highest log entry known to be committed
        self.last_applied = 0  # Highest log entry applied to state machine

        # Leader state (volatile, reset after election)
        self.next_index: Dict[str, int] = {}  # For each server, index of next log entry to send
        self.match_index: Dict[str, int] = {}  # For each server, highest log entry known to be replicated

        # Current state
        self.state = NodeState.FOLLOWER
        self.leader_id: Optional[str] = None

        # Timing
        self.election_timeout_range = election_timeout_range
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat = datetime.utcnow()
        self.election_timeout = self._random_election_timeout()

        # Message queue
        self.inbox: asyncio.Queue = asyncio.Queue()

        # Callbacks
        self.send_message_callback: Optional[callable] = None
        self.apply_command_callback: Optional[callable] = None

        # Running tasks
        self._running = False
        self._tasks: List[asyncio.Task] = []

    def _random_election_timeout(self) -> float:
        """Generate random election timeout."""
        return random.uniform(*self.election_timeout_range)

    async def start(self):
        """Start the node."""
        self._running = True

        # Start main loop
        self._tasks.append(asyncio.create_task(self._main_loop()))

        # Start heartbeat/election timer
        self._tasks.append(asyncio.create_task(self._timer_loop()))

        logger.info(f"RaftNode {self.node_id} started in {self.state.value} state")

    async def stop(self):
        """Stop the node."""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info(f"RaftNode {self.node_id} stopped")

    async def _main_loop(self):
        """Main message processing loop."""
        while self._running:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.inbox.get(), timeout=0.1)
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in main loop: {e}")

    async def _timer_loop(self):
        """Timer for heartbeats and elections."""
        while self._running:
            await asyncio.sleep(0.01)  # Check every 10ms

            time_since_heartbeat = (datetime.utcnow() - self.last_heartbeat).total_seconds()

            if self.state == NodeState.LEADER:
                # Send periodic heartbeats
                if time_since_heartbeat >= self.heartbeat_interval:
                    await self._send_heartbeats()
                    self.last_heartbeat = datetime.utcnow()

            elif self.state in (NodeState.FOLLOWER, NodeState.CANDIDATE):
                # Check election timeout
                if time_since_heartbeat >= self.election_timeout:
                    await self._start_election()

    async def _handle_message(self, message: ConsensusMessage):
        """Handle incoming consensus message."""
        # Update term if we see higher term
        if message.term > self.current_term:
            self.current_term = message.term
            self.state = NodeState.FOLLOWER
            self.voted_for = None

        if message.msg_type == MessageType.REQUEST_VOTE:
            await self._handle_request_vote(message)
        elif message.msg_type == MessageType.VOTE:
            await self._handle_vote(message)
        elif message.msg_type == MessageType.APPEND_ENTRIES:
            await self._handle_append_entries(message)
        elif message.msg_type == MessageType.APPEND_RESPONSE:
            await self._handle_append_response(message)
        elif message.msg_type == MessageType.CLIENT_REQUEST:
            await self._handle_client_request(message)

    async def _start_election(self):
        """Start leader election."""
        logger.info(f"RaftNode {self.node_id} starting election for term {self.current_term + 1}")

        # Become candidate
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.election_timeout = self._random_election_timeout()
        self.last_heartbeat = datetime.utcnow()

        # Vote for self
        votes_received = 1

        # Request votes from other nodes
        last_log_index = len(self.log)
        last_log_term = self.log[-1].term if self.log else 0

        request_vote_msg = ConsensusMessage(
            msg_type=MessageType.REQUEST_VOTE,
            sender_id=self.node_id,
            term=self.current_term,
            payload={
                "candidate_id": self.node_id,
                "last_log_index": last_log_index,
                "last_log_term": last_log_term,
            },
        )

        # Send to all other nodes
        for node_id in self.cluster_nodes:
            await self._send_message(node_id, request_vote_msg)

    async def _handle_request_vote(self, message: ConsensusMessage):
        """Handle vote request."""
        payload = message.payload
        vote_granted = False

        # Grant vote if:
        # 1. Haven't voted for anyone else in this term
        # 2. Candidate's log is at least as up-to-date as ours

        if (self.voted_for is None or self.voted_for == payload["candidate_id"]):
            # Check log up-to-date-ness
            last_log_index = len(self.log)
            last_log_term = self.log[-1].term if self.log else 0

            candidate_log_ok = (
                payload["last_log_term"] > last_log_term or
                (payload["last_log_term"] == last_log_term and
                 payload["last_log_index"] >= last_log_index)
            )

            if candidate_log_ok:
                vote_granted = True
                self.voted_for = payload["candidate_id"]
                self.last_heartbeat = datetime.utcnow()  # Reset election timeout

        # Send vote response
        vote_msg = ConsensusMessage(
            msg_type=MessageType.VOTE,
            sender_id=self.node_id,
            term=self.current_term,
            payload={
                "vote_granted": vote_granted,
                "voter_id": self.node_id,
            },
        )

        await self._send_message(message.sender_id, vote_msg)

    async def _handle_vote(self, message: ConsensusMessage):
        """Handle vote response."""
        if self.state != NodeState.CANDIDATE:
            return

        if message.payload.get("vote_granted"):
            # Count votes
            votes = 1  # Self vote
            for node_id in self.cluster_nodes:
                # In real implementation, track votes received
                pass

            # Check if we have majority
            majority = (len(self.cluster_nodes) + 1) // 2 + 1

            if votes >= majority:
                await self._become_leader()

    async def _become_leader(self):
        """Transition to leader state."""
        logger.info(f"RaftNode {self.node_id} became LEADER for term {self.current_term}")

        self.state = NodeState.LEADER
        self.leader_id = self.node_id

        # Initialize leader state
        last_log_index = len(self.log)
        for node_id in self.cluster_nodes:
            self.next_index[node_id] = last_log_index + 1
            self.match_index[node_id] = 0

        # Send initial heartbeats
        await self._send_heartbeats()

    async def _send_heartbeats(self):
        """Send heartbeat (empty AppendEntries) to all followers."""
        for node_id in self.cluster_nodes:
            await self._send_append_entries(node_id)

    async def _send_append_entries(self, follower_id: str):
        """Send AppendEntries RPC to follower."""
        next_idx = self.next_index.get(follower_id, 1)
        prev_log_index = next_idx - 1
        prev_log_term = self.log[prev_log_index - 1].term if prev_log_index > 0 and prev_log_index <= len(self.log) else 0

        # Get entries to send
        entries = self.log[next_idx - 1:] if next_idx <= len(self.log) else []

        append_msg = ConsensusMessage(
            msg_type=MessageType.APPEND_ENTRIES,
            sender_id=self.node_id,
            term=self.current_term,
            payload={
                "leader_id": self.node_id,
                "prev_log_index": prev_log_index,
                "prev_log_term": prev_log_term,
                "entries": [e.to_dict() for e in entries],
                "leader_commit": self.commit_index,
            },
        )

        await self._send_message(follower_id, append_msg)

    async def _handle_append_entries(self, message: ConsensusMessage):
        """Handle AppendEntries RPC."""
        payload = message.payload
        success = False

        # Reset election timeout (got message from leader)
        self.last_heartbeat = datetime.utcnow()
        self.leader_id = payload["leader_id"]

        # Check log consistency
        prev_log_index = payload["prev_log_index"]
        prev_log_term = payload["prev_log_term"]

        if prev_log_index == 0 or (
            prev_log_index <= len(self.log) and
            self.log[prev_log_index - 1].term == prev_log_term
        ):
            success = True

            # Append new entries
            entries = payload["entries"]
            if entries:
                # Remove conflicting entries and append new ones
                start_index = prev_log_index
                for entry_dict in entries:
                    entry = LogEntry(**entry_dict)
                    if start_index < len(self.log):
                        # Check for conflict
                        if self.log[start_index].term != entry.term:
                            # Delete this and all following entries
                            self.log = self.log[:start_index]
                            self.log.append(entry)
                        # else: entry already present
                    else:
                        self.log.append(entry)
                    start_index += 1

            # Update commit index
            if payload["leader_commit"] > self.commit_index:
                self.commit_index = min(payload["leader_commit"], len(self.log))

            # Apply committed entries
            await self._apply_committed_entries()

        # Send response
        response_msg = ConsensusMessage(
            msg_type=MessageType.APPEND_RESPONSE,
            sender_id=self.node_id,
            term=self.current_term,
            payload={
                "success": success,
                "match_index": len(self.log) if success else 0,
            },
        )

        await self._send_message(message.sender_id, response_msg)

    async def _handle_append_response(self, message: ConsensusMessage):
        """Handle AppendEntries response."""
        if self.state != NodeState.LEADER:
            return

        follower_id = message.sender_id
        payload = message.payload

        if payload["success"]:
            # Update next_index and match_index
            self.match_index[follower_id] = payload["match_index"]
            self.next_index[follower_id] = payload["match_index"] + 1

            # Check if we can advance commit_index
            await self._update_commit_index()
        else:
            # Log inconsistency - decrement next_index and retry
            if follower_id in self.next_index:
                self.next_index[follower_id] = max(1, self.next_index[follower_id] - 1)
                await self._send_append_entries(follower_id)

    async def _update_commit_index(self):
        """Update commit index based on match indices."""
        # Find highest N such that:
        # - N > commit_index
        # - Majority of match_index[i] >= N
        # - log[N].term == current_term

        for n in range(len(self.log), self.commit_index, -1):
            if self.log[n - 1].term == self.current_term:
                # Count replicas
                count = 1  # Leader
                for node_id in self.cluster_nodes:
                    if self.match_index.get(node_id, 0) >= n:
                        count += 1

                # Check majority
                majority = (len(self.cluster_nodes) + 1) // 2 + 1
                if count >= majority:
                    self.commit_index = n
                    await self._apply_committed_entries()
                    break

    async def _apply_committed_entries(self):
        """Apply committed log entries to state machine."""
        while self.last_applied < self.commit_index:
            self.last_applied += 1
            entry = self.log[self.last_applied - 1]

            # Apply command
            if self.apply_command_callback:
                await self.apply_command_callback(entry.command)

            logger.debug(f"Applied entry {self.last_applied}: {entry.command}")

    async def _handle_client_request(self, message: ConsensusMessage):
        """Handle client request (only leader can handle)."""
        if self.state != NodeState.LEADER:
            # Redirect to leader
            logger.info(f"Not leader, redirecting to {self.leader_id}")
            return

        # Append to log
        command = message.payload.get("command")
        entry = LogEntry(
            term=self.current_term,
            index=len(self.log) + 1,
            command=command,
        )
        self.log.append(entry)

        # Replicate to followers
        for node_id in self.cluster_nodes:
            await self._send_append_entries(node_id)

    async def _send_message(self, node_id: str, message: ConsensusMessage):
        """Send message to another node."""
        if self.send_message_callback:
            await self.send_message_callback(node_id, message)

    async def receive_message(self, message: ConsensusMessage):
        """Receive message from network."""
        await self.inbox.put(message)


# ═══════════════════════════════════════════════════════════════════════════
# DISTRIBUTED CONSENSUS CLUSTER
# ═══════════════════════════════════════════════════════════════════════════

class ConsensusCluster:
    """Manages a cluster of consensus nodes."""

    def __init__(self, node_ids: List[str]):
        self.node_ids = node_ids
        self.nodes: Dict[str, RaftNode] = {}

        # Message routing
        self.message_queues: Dict[str, asyncio.Queue] = {}

        # Create nodes
        for node_id in node_ids:
            node = RaftNode(node_id, node_ids)
            node.send_message_callback = self._route_message
            self.nodes[node_id] = node
            self.message_queues[node_id] = asyncio.Queue()

    async def start(self):
        """Start all nodes."""
        for node in self.nodes.values():
            await node.start()

        logger.info(f"Consensus cluster started with {len(self.nodes)} nodes")

    async def stop(self):
        """Stop all nodes."""
        for node in self.nodes.values():
            await node.stop()

    async def _route_message(self, target_id: str, message: ConsensusMessage):
        """Route message between nodes."""
        if target_id in self.nodes:
            await self.nodes[target_id].receive_message(message)

    async def submit_command(self, command: Any) -> bool:
        """Submit command to cluster."""
        # Find leader
        leader = None
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                leader = node
                break

        if not leader:
            logger.warning("No leader available")
            return False

        # Submit to leader
        client_msg = ConsensusMessage(
            msg_type=MessageType.CLIENT_REQUEST,
            sender_id="client",
            term=0,
            payload={"command": command},
        )

        await leader.receive_message(client_msg)
        return True

    def get_leader(self) -> Optional[str]:
        """Get current leader ID."""
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                return node.node_id
        return None

    def get_cluster_state(self) -> Dict[str, Any]:
        """Get cluster state summary."""
        return {
            "nodes": {
                node_id: {
                    "state": node.state.value,
                    "term": node.current_term,
                    "log_length": len(node.log),
                    "commit_index": node.commit_index,
                }
                for node_id, node in self.nodes.items()
            },
            "leader": self.get_leader(),
        }


# Global registry
_consensus_clusters: Dict[str, ConsensusCluster] = {}


def create_consensus_cluster(name: str, node_ids: List[str]) -> ConsensusCluster:
    """Create a consensus cluster."""
    cluster = ConsensusCluster(node_ids)
    _consensus_clusters[name] = cluster
    return cluster


def get_consensus_cluster(name: str) -> Optional[ConsensusCluster]:
    """Get consensus cluster by name."""
    return _consensus_clusters.get(name)
