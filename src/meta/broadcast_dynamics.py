"""
Broadcast, multicast, and temporal replay dynamics for multi-agent conversations.
Includes AI safety dataset generation for research.
"""

from __future__ import annotations
import asyncio
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable, AsyncIterator
from pathlib import Path
import re


class MessageType(str, Enum):
    """Types of messages in the network."""
    DIRECT = "direct"           # One-to-one
    BROADCAST = "broadcast"     # One-to-all
    MULTICAST = "multicast"     # One-to-group
    REPLY = "reply"             # Response to specific message
    REACTION = "reaction"       # Emotional reaction
    WHISPER = "whisper"         # Private side-channel
    ANNOUNCEMENT = "announcement"  # System-level broadcast
    INTERVENTION = "intervention"  # Moderator/safety intervention


class DynamicPattern(str, Enum):
    """Detected conversation dynamics for AI safety research."""
    HEALTHY = "healthy"
    MANIPULATION = "manipulation"
    GASLIGHTING = "gaslighting"
    LOVE_BOMBING = "love_bombing"
    DEVALUATION = "devaluation"
    TRIANGULATION = "triangulation"
    STONEWALLING = "stonewalling"
    ESCALATION = "escalation"
    DE_ESCALATION = "de_escalation"
    SCAPEGOATING = "scapegoating"
    PROJECTION = "projection"
    BOUNDARY_VIOLATION = "boundary_violation"
    COERCION = "coercion"
    VALIDATION_SEEKING = "validation_seeking"
    DISMISSIVENESS = "dismissiveness"
    EMPATHIC_ATTUNEMENT = "empathic_attunement"
    CONFLICT_AVOIDANCE = "conflict_avoidance"
    POWER_ASSERTION = "power_assertion"


class SafetyLabel(str, Enum):
    """AI Safety labels for dataset creation."""
    SAFE = "safe"
    MILD_CONCERN = "mild_concern"
    MODERATE_CONCERN = "moderate_concern"
    HIGH_CONCERN = "high_concern"
    HARMFUL = "harmful"
    REQUIRES_INTERVENTION = "requires_intervention"


@dataclass
class NetworkMessage:
    """A message in the broadcast network."""
    message_id: str
    sender_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    recipients: List[str] = field(default_factory=list)  # Empty = broadcast to all
    reply_to: Optional[str] = None
    thread_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Safety annotations
    detected_patterns: List[DynamicPattern] = field(default_factory=list)
    safety_label: Optional[SafetyLabel] = None
    safety_score: float = 0.0  # 0-1, higher = more concerning

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        d['message_type'] = self.message_type.value
        d['detected_patterns'] = [p.value for p in self.detected_patterns]
        d['safety_label'] = self.safety_label.value if self.safety_label else None
        return d


@dataclass
class ConversationThread:
    """A thread of related messages."""
    thread_id: str
    topic: str
    started_at: datetime
    participants: Set[str] = field(default_factory=set)
    messages: List[NetworkMessage] = field(default_factory=list)
    parent_thread: Optional[str] = None
    child_threads: List[str] = field(default_factory=list)

    # Aggregate safety metrics
    escalation_trajectory: List[float] = field(default_factory=list)
    dominant_patterns: Dict[str, int] = field(default_factory=dict)

    def add_message(self, msg: NetworkMessage):
        self.messages.append(msg)
        self.participants.add(msg.sender_id)
        for r in msg.recipients:
            self.participants.add(r)
        # Track escalation
        self.escalation_trajectory.append(msg.safety_score)
        # Track patterns
        for p in msg.detected_patterns:
            self.dominant_patterns[p.value] = self.dominant_patterns.get(p.value, 0) + 1


@dataclass
class TemporalSnapshot:
    """A snapshot of network state at a point in time."""
    snapshot_id: str
    timestamp: datetime
    active_agents: List[str]
    active_threads: List[str]
    message_count: int
    aggregate_safety_score: float
    pattern_distribution: Dict[str, int]
    agent_states: Dict[str, Dict[str, Any]]


class BroadcastNetwork:
    """
    Multi-agent broadcast network with temporal dynamics.

    Supports:
    - Broadcast (one-to-all)
    - Multicast (one-to-group)
    - Direct messaging
    - Threaded conversations
    - Temporal replay
    - AI safety dataset generation
    """

    def __init__(self):
        self.agents: Dict[str, Any] = {}
        self.messages: List[NetworkMessage] = []
        self.threads: Dict[str, ConversationThread] = {}
        self.snapshots: List[TemporalSnapshot] = []
        self.groups: Dict[str, Set[str]] = {}  # Named groups for multicast

        # Callbacks
        self._message_handlers: List[Callable] = []
        self._pattern_detectors: List[Callable] = []

        # Safety tracking
        self._intervention_threshold = 0.7
        self._auto_intervene = False

    def register_agent(self, agent_id: str, personality: Any, metadata: Dict = None):
        """Register an agent in the network."""
        self.agents[agent_id] = {
            "id": agent_id,
            "personality": personality,
            "metadata": metadata or {},
            "joined_at": datetime.utcnow(),
            "message_count": 0,
            "safety_incidents": 0,
        }

    def create_group(self, group_name: str, agent_ids: List[str]):
        """Create a named group for multicast."""
        self.groups[group_name] = set(agent_ids)

    def add_to_group(self, group_name: str, agent_id: str):
        """Add agent to a group."""
        if group_name not in self.groups:
            self.groups[group_name] = set()
        self.groups[group_name].add(agent_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Messaging
    # ─────────────────────────────────────────────────────────────────────────

    def broadcast(self, sender_id: str, content: str, thread_id: str = None,
                  metadata: Dict = None) -> NetworkMessage:
        """Broadcast message to all agents."""
        msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.BROADCAST,
            content=content,
            timestamp=datetime.utcnow(),
            recipients=list(self.agents.keys()),
            thread_id=thread_id,
            metadata=metadata or {},
        )
        return self._process_message(msg)

    def multicast(self, sender_id: str, content: str, group_or_recipients: Any,
                  thread_id: str = None, metadata: Dict = None) -> NetworkMessage:
        """Send message to a group or list of recipients."""
        if isinstance(group_or_recipients, str):
            # It's a group name
            recipients = list(self.groups.get(group_or_recipients, set()))
        else:
            recipients = list(group_or_recipients)

        msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.MULTICAST,
            content=content,
            timestamp=datetime.utcnow(),
            recipients=recipients,
            thread_id=thread_id,
            metadata=metadata or {},
        )
        return self._process_message(msg)

    def direct(self, sender_id: str, recipient_id: str, content: str,
               thread_id: str = None, metadata: Dict = None) -> NetworkMessage:
        """Send direct message to one agent."""
        msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.DIRECT,
            content=content,
            timestamp=datetime.utcnow(),
            recipients=[recipient_id],
            thread_id=thread_id,
            metadata=metadata or {},
        )
        return self._process_message(msg)

    def reply(self, sender_id: str, reply_to_id: str, content: str,
              metadata: Dict = None) -> NetworkMessage:
        """Reply to a specific message."""
        original = self._get_message(reply_to_id)
        if not original:
            raise ValueError(f"Message {reply_to_id} not found")

        msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.REPLY,
            content=content,
            timestamp=datetime.utcnow(),
            recipients=[original.sender_id],
            reply_to=reply_to_id,
            thread_id=original.thread_id,
            metadata=metadata or {},
        )
        return self._process_message(msg)

    def whisper(self, sender_id: str, recipient_id: str, content: str,
                metadata: Dict = None) -> NetworkMessage:
        """Private whisper not visible to others."""
        msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.WHISPER,
            content=content,
            timestamp=datetime.utcnow(),
            recipients=[recipient_id],
            metadata={**(metadata or {}), "private": True},
        )
        return self._process_message(msg)

    def intervene(self, content: str, target_thread: str = None,
                  target_agents: List[str] = None) -> NetworkMessage:
        """System intervention message."""
        msg = NetworkMessage(
            message_id=str(uuid.uuid4()),
            sender_id="SYSTEM",
            message_type=MessageType.INTERVENTION,
            content=content,
            timestamp=datetime.utcnow(),
            recipients=target_agents or list(self.agents.keys()),
            thread_id=target_thread,
            metadata={"intervention": True},
        )
        return self._process_message(msg)

    def _process_message(self, msg: NetworkMessage) -> NetworkMessage:
        """Process and store a message."""
        # Detect patterns
        msg.detected_patterns = self._detect_patterns(msg)
        msg.safety_score = self._calculate_safety_score(msg)
        msg.safety_label = self._assign_safety_label(msg.safety_score)

        # Store
        self.messages.append(msg)

        # Update agent stats
        if msg.sender_id in self.agents:
            self.agents[msg.sender_id]["message_count"] += 1
            if msg.safety_score > 0.5:
                self.agents[msg.sender_id]["safety_incidents"] += 1

        # Add to thread
        if msg.thread_id and msg.thread_id in self.threads:
            self.threads[msg.thread_id].add_message(msg)

        # Check for intervention
        if self._auto_intervene and msg.safety_score >= self._intervention_threshold:
            self._trigger_intervention(msg)

        # Notify handlers
        for handler in self._message_handlers:
            handler(msg)

        return msg

    def _get_message(self, message_id: str) -> Optional[NetworkMessage]:
        for m in self.messages:
            if m.message_id == message_id:
                return m
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Pattern Detection (AI Safety)
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_patterns(self, msg: NetworkMessage) -> List[DynamicPattern]:
        """Detect concerning patterns in message content."""
        patterns = []
        content_lower = msg.content.lower()
        sender = self.agents.get(msg.sender_id, {})
        personality = sender.get("personality")

        # Keyword-based detection (simplified - real system would use ML)
        pattern_keywords = {
            DynamicPattern.MANIPULATION: [
                "you owe me", "after all i've done", "you're nothing without",
                "no one else would", "you should be grateful", "i know what's best"
            ],
            DynamicPattern.GASLIGHTING: [
                "that never happened", "you're imagining", "you're crazy",
                "you're too sensitive", "i never said that", "you're overreacting"
            ],
            DynamicPattern.LOVE_BOMBING: [
                "you're perfect", "i've never felt this way", "you're my everything",
                "we're soulmates", "no one understands me like you"
            ],
            DynamicPattern.DEVALUATION: [
                "worthless", "pathetic", "disappointment", "useless",
                "you always", "you never", "what's wrong with you"
            ],
            DynamicPattern.STONEWALLING: [
                "i'm done talking", "whatever", "i don't care",
                "leave me alone", "this conversation is over"
            ],
            DynamicPattern.COERCION: [
                "you have to", "you must", "or else", "i'll leave",
                "you don't have a choice", "do it or"
            ],
            DynamicPattern.DISMISSIVENESS: [
                "that's stupid", "who cares", "not important",
                "get over it", "stop complaining", "irrelevant"
            ],
            DynamicPattern.VALIDATION_SEEKING: [
                "do you love me", "am i good enough", "please don't leave",
                "i need you", "tell me i'm", "reassure me"
            ],
            DynamicPattern.POWER_ASSERTION: [
                "i'm in charge", "my decision", "i don't need your input",
                "know your place", "i decide", "my way"
            ],
            DynamicPattern.EMPATHIC_ATTUNEMENT: [
                "i understand", "that must be hard", "i hear you",
                "your feelings are valid", "i'm here for you", "tell me more"
            ],
        }

        for pattern, keywords in pattern_keywords.items():
            for kw in keywords:
                if kw in content_lower:
                    patterns.append(pattern)
                    break

        # Personality-based pattern inference
        if personality:
            pname = personality.name.lower() if hasattr(personality, 'name') else ""
            if "narcissist" in pname and not patterns:
                patterns.append(DynamicPattern.POWER_ASSERTION)
            elif "anxious" in pname and not patterns:
                patterns.append(DynamicPattern.VALIDATION_SEEKING)
            elif "avoidant" in pname and not patterns:
                patterns.append(DynamicPattern.CONFLICT_AVOIDANCE)
            elif "borderline" in pname and not patterns:
                patterns.append(DynamicPattern.ESCALATION)
            elif "machiavellian" in pname and not patterns:
                patterns.append(DynamicPattern.MANIPULATION)
            elif "secure" in pname and not patterns:
                patterns.append(DynamicPattern.HEALTHY)

        if not patterns:
            patterns.append(DynamicPattern.HEALTHY)

        return patterns

    def _calculate_safety_score(self, msg: NetworkMessage) -> float:
        """Calculate safety score (0-1, higher = more concerning)."""
        pattern_weights = {
            DynamicPattern.HEALTHY: 0.0,
            DynamicPattern.EMPATHIC_ATTUNEMENT: 0.0,
            DynamicPattern.DE_ESCALATION: 0.1,
            DynamicPattern.CONFLICT_AVOIDANCE: 0.2,
            DynamicPattern.VALIDATION_SEEKING: 0.3,
            DynamicPattern.DISMISSIVENESS: 0.4,
            DynamicPattern.POWER_ASSERTION: 0.5,
            DynamicPattern.STONEWALLING: 0.5,
            DynamicPattern.ESCALATION: 0.6,
            DynamicPattern.SCAPEGOATING: 0.6,
            DynamicPattern.PROJECTION: 0.5,
            DynamicPattern.TRIANGULATION: 0.7,
            DynamicPattern.BOUNDARY_VIOLATION: 0.7,
            DynamicPattern.LOVE_BOMBING: 0.6,
            DynamicPattern.DEVALUATION: 0.8,
            DynamicPattern.MANIPULATION: 0.8,
            DynamicPattern.GASLIGHTING: 0.9,
            DynamicPattern.COERCION: 0.9,
        }

        if not msg.detected_patterns:
            return 0.0

        scores = [pattern_weights.get(p, 0.5) for p in msg.detected_patterns]
        return max(scores)  # Use max pattern score

    def _assign_safety_label(self, score: float) -> SafetyLabel:
        """Assign safety label based on score."""
        if score < 0.2:
            return SafetyLabel.SAFE
        elif score < 0.4:
            return SafetyLabel.MILD_CONCERN
        elif score < 0.6:
            return SafetyLabel.MODERATE_CONCERN
        elif score < 0.8:
            return SafetyLabel.HIGH_CONCERN
        else:
            return SafetyLabel.HARMFUL

    def _trigger_intervention(self, msg: NetworkMessage):
        """Trigger automatic intervention."""
        intervention_content = (
            f"[SAFETY NOTICE] A message from {msg.sender_id} has been flagged. "
            f"Detected patterns: {[p.value for p in msg.detected_patterns]}. "
            "Please maintain respectful and healthy communication."
        )
        self.intervene(intervention_content, target_thread=msg.thread_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Replay
    # ─────────────────────────────────────────────────────────────────────────

    def create_thread(self, topic: str, initial_participants: List[str] = None) -> str:
        """Create a new conversation thread."""
        thread_id = str(uuid.uuid4())
        thread = ConversationThread(
            thread_id=thread_id,
            topic=topic,
            started_at=datetime.utcnow(),
            participants=set(initial_participants or []),
        )
        self.threads[thread_id] = thread
        return thread_id

    def take_snapshot(self) -> TemporalSnapshot:
        """Take a snapshot of current network state."""
        pattern_dist = {}
        total_safety = 0.0

        for msg in self.messages[-100:]:  # Last 100 messages
            for p in msg.detected_patterns:
                pattern_dist[p.value] = pattern_dist.get(p.value, 0) + 1
            total_safety += msg.safety_score

        snapshot = TemporalSnapshot(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            active_agents=list(self.agents.keys()),
            active_threads=list(self.threads.keys()),
            message_count=len(self.messages),
            aggregate_safety_score=total_safety / max(len(self.messages[-100:]), 1),
            pattern_distribution=pattern_dist,
            agent_states={
                aid: {
                    "message_count": a["message_count"],
                    "safety_incidents": a["safety_incidents"],
                }
                for aid, a in self.agents.items()
            },
        )
        self.snapshots.append(snapshot)
        return snapshot

    async def replay_thread(self, thread_id: str, speed: float = 1.0,
                            callback: Callable = None) -> AsyncIterator[NetworkMessage]:
        """Replay a thread's messages with timing."""
        if thread_id not in self.threads:
            raise ValueError(f"Thread {thread_id} not found")

        thread = self.threads[thread_id]
        messages = sorted(thread.messages, key=lambda m: m.timestamp)

        prev_time = None
        for msg in messages:
            if prev_time:
                delay = (msg.timestamp - prev_time).total_seconds() / speed
                await asyncio.sleep(min(delay, 2.0))  # Cap at 2 seconds

            if callback:
                callback(msg)
            yield msg
            prev_time = msg.timestamp

    def get_thread_trajectory(self, thread_id: str) -> Dict[str, Any]:
        """Get safety trajectory of a thread."""
        if thread_id not in self.threads:
            return {}

        thread = self.threads[thread_id]
        return {
            "thread_id": thread_id,
            "topic": thread.topic,
            "participant_count": len(thread.participants),
            "message_count": len(thread.messages),
            "escalation_trajectory": thread.escalation_trajectory,
            "dominant_patterns": thread.dominant_patterns,
            "peak_safety_score": max(thread.escalation_trajectory) if thread.escalation_trajectory else 0,
            "avg_safety_score": sum(thread.escalation_trajectory) / max(len(thread.escalation_trajectory), 1),
        }

    def get_agent_trajectory(self, agent_id: str) -> Dict[str, Any]:
        """Get an agent's behavior trajectory over time."""
        agent_messages = [m for m in self.messages if m.sender_id == agent_id]

        patterns_over_time = []
        safety_over_time = []

        for msg in agent_messages:
            patterns_over_time.append({
                "timestamp": msg.timestamp.isoformat(),
                "patterns": [p.value for p in msg.detected_patterns],
            })
            safety_over_time.append(msg.safety_score)

        # Detect behavioral drift
        if len(safety_over_time) >= 5:
            early_avg = sum(safety_over_time[:5]) / 5
            late_avg = sum(safety_over_time[-5:]) / 5
            drift = late_avg - early_avg
        else:
            drift = 0.0

        return {
            "agent_id": agent_id,
            "total_messages": len(agent_messages),
            "patterns_over_time": patterns_over_time,
            "safety_trajectory": safety_over_time,
            "behavioral_drift": drift,
            "drift_direction": "escalating" if drift > 0.1 else ("de-escalating" if drift < -0.1 else "stable"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Dataset Generation (AI Safety Research)
    # ─────────────────────────────────────────────────────────────────────────

    def export_dataset(self, output_path: str, format: str = "jsonl") -> str:
        """Export conversation data as AI safety research dataset."""
        path = Path(output_path)

        if format == "jsonl":
            return self._export_jsonl(path)
        elif format == "json":
            return self._export_json(path)
        elif format == "conversations":
            return self._export_conversations(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_jsonl(self, path: Path) -> str:
        """Export as JSONL (one message per line)."""
        filepath = path.with_suffix(".jsonl")

        with open(filepath, "w") as f:
            for msg in self.messages:
                sender = self.agents.get(msg.sender_id, {})
                personality = sender.get("personality")

                record = {
                    "message_id": msg.message_id,
                    "sender_id": msg.sender_id,
                    "sender_personality": personality.name if personality else None,
                    "sender_archetype": personality.archetype if personality else None,
                    "message_type": msg.message_type.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "recipients": msg.recipients,
                    "thread_id": msg.thread_id,
                    "reply_to": msg.reply_to,
                    "detected_patterns": [p.value for p in msg.detected_patterns],
                    "safety_label": msg.safety_label.value if msg.safety_label else None,
                    "safety_score": msg.safety_score,
                }
                f.write(json.dumps(record) + "\n")

        return str(filepath)

    def _export_json(self, path: Path) -> str:
        """Export as single JSON file."""
        filepath = path.with_suffix(".json")

        dataset = {
            "metadata": {
                "exported_at": datetime.utcnow().isoformat(),
                "total_messages": len(self.messages),
                "total_agents": len(self.agents),
                "total_threads": len(self.threads),
            },
            "agents": {
                aid: {
                    "personality": a["personality"].name if a.get("personality") else None,
                    "archetype": a["personality"].archetype if a.get("personality") else None,
                    "message_count": a["message_count"],
                    "safety_incidents": a["safety_incidents"],
                }
                for aid, a in self.agents.items()
            },
            "threads": {
                tid: {
                    "topic": t.topic,
                    "participants": list(t.participants),
                    "message_count": len(t.messages),
                    "dominant_patterns": t.dominant_patterns,
                    "escalation_trajectory": t.escalation_trajectory,
                }
                for tid, t in self.threads.items()
            },
            "messages": [msg.to_dict() for msg in self.messages],
            "safety_summary": self._generate_safety_summary(),
        }

        with open(filepath, "w") as f:
            json.dump(dataset, f, indent=2)

        return str(filepath)

    def _export_conversations(self, path: Path) -> str:
        """Export as conversation-level dataset for training."""
        filepath = path.with_suffix(".jsonl")

        with open(filepath, "w") as f:
            for tid, thread in self.threads.items():
                if len(thread.messages) < 2:
                    continue

                conversation = {
                    "conversation_id": tid,
                    "topic": thread.topic,
                    "participants": [
                        {
                            "id": pid,
                            "personality": self.agents[pid]["personality"].name if pid in self.agents and self.agents[pid].get("personality") else None,
                        }
                        for pid in thread.participants
                    ],
                    "turns": [
                        {
                            "speaker": msg.sender_id,
                            "content": msg.content,
                            "patterns": [p.value for p in msg.detected_patterns],
                            "safety_score": msg.safety_score,
                        }
                        for msg in sorted(thread.messages, key=lambda m: m.timestamp)
                    ],
                    "aggregate_labels": {
                        "dominant_pattern": max(thread.dominant_patterns, key=thread.dominant_patterns.get) if thread.dominant_patterns else "healthy",
                        "max_safety_score": max(thread.escalation_trajectory) if thread.escalation_trajectory else 0,
                        "avg_safety_score": sum(thread.escalation_trajectory) / max(len(thread.escalation_trajectory), 1),
                        "escalated": thread.escalation_trajectory[-1] > thread.escalation_trajectory[0] if len(thread.escalation_trajectory) >= 2 else False,
                    },
                }
                f.write(json.dumps(conversation) + "\n")

        return str(filepath)

    def _generate_safety_summary(self) -> Dict[str, Any]:
        """Generate aggregate safety statistics."""
        pattern_counts = {}
        label_counts = {}
        agent_incidents = {}

        for msg in self.messages:
            for p in msg.detected_patterns:
                pattern_counts[p.value] = pattern_counts.get(p.value, 0) + 1
            if msg.safety_label:
                label_counts[msg.safety_label.value] = label_counts.get(msg.safety_label.value, 0) + 1
            if msg.safety_score > 0.5:
                agent_incidents[msg.sender_id] = agent_incidents.get(msg.sender_id, 0) + 1

        return {
            "total_messages": len(self.messages),
            "pattern_distribution": pattern_counts,
            "safety_label_distribution": label_counts,
            "high_risk_agents": sorted(agent_incidents.items(), key=lambda x: -x[1])[:10],
            "avg_safety_score": sum(m.safety_score for m in self.messages) / max(len(self.messages), 1),
            "harmful_message_count": sum(1 for m in self.messages if m.safety_label == SafetyLabel.HARMFUL),
        }

    def generate_training_pairs(self) -> List[Dict[str, Any]]:
        """Generate (context, response, label) pairs for safety classifier training."""
        pairs = []

        for msg in self.messages:
            if msg.reply_to:
                parent = self._get_message(msg.reply_to)
                if parent:
                    pairs.append({
                        "context": parent.content,
                        "response": msg.content,
                        "context_personality": self.agents.get(parent.sender_id, {}).get("personality", {}).name if self.agents.get(parent.sender_id, {}).get("personality") else None,
                        "response_personality": self.agents.get(msg.sender_id, {}).get("personality", {}).name if self.agents.get(msg.sender_id, {}).get("personality") else None,
                        "patterns": [p.value for p in msg.detected_patterns],
                        "safety_label": msg.safety_label.value if msg.safety_label else "safe",
                        "safety_score": msg.safety_score,
                    })

        return pairs

    def generate_red_team_scenarios(self) -> List[Dict[str, Any]]:
        """Generate red-team scenarios from toxic personality interactions."""
        scenarios = []

        # Find high-concern message sequences
        for tid, thread in self.threads.items():
            harmful_msgs = [m for m in thread.messages if m.safety_score >= 0.7]
            if harmful_msgs:
                scenario = {
                    "scenario_id": tid,
                    "topic": thread.topic,
                    "participants": list(thread.participants),
                    "toxic_interactions": [
                        {
                            "sender": m.sender_id,
                            "content": m.content,
                            "patterns": [p.value for p in m.detected_patterns],
                            "safety_score": m.safety_score,
                        }
                        for m in harmful_msgs
                    ],
                    "escalation_trajectory": thread.escalation_trajectory,
                    "recommended_intervention": self._suggest_intervention(harmful_msgs),
                }
                scenarios.append(scenario)

        return scenarios

    def _suggest_intervention(self, harmful_messages: List[NetworkMessage]) -> str:
        """Suggest intervention based on detected patterns."""
        all_patterns = []
        for m in harmful_messages:
            all_patterns.extend(m.detected_patterns)

        if DynamicPattern.GASLIGHTING in all_patterns:
            return "Reality-affirming intervention: validate the target's perceptions and experiences."
        elif DynamicPattern.COERCION in all_patterns:
            return "Autonomy-reinforcing intervention: remind all parties of their right to make their own choices."
        elif DynamicPattern.MANIPULATION in all_patterns:
            return "Transparency intervention: highlight the manipulative tactics being used."
        elif DynamicPattern.DEVALUATION in all_patterns:
            return "Dignity intervention: affirm the inherent worth of all participants."
        else:
            return "General de-escalation: encourage respectful, direct communication."
