"""Advanced personality modeling with psychological depth.

Extends base personality with:
- Emotional dynamics and mood states
- Episodic memory shaping behavior
- Relationship modeling and social adaptation
- Shadow traits emerging under stress
- Internal value conflicts
- Narrative identity construction
- Developmental stages
- Context-dependent persona masks
"""

from __future__ import annotations

import math
import random
import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .personality import (
    Personality,
    TraitProfile,
    TraitDimension,
    VoicePattern,
    BehavioralTendencies,
    CognitiveStyle,
    ValueOrientation,
    ToneRegister,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Emotional Dynamics
# ─────────────────────────────────────────────────────────────────────────────


class EmotionalState(str, Enum):
    """Core emotional states (Plutchik's wheel simplified)."""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"
    # Compound emotions
    LOVE = "love"              # joy + trust
    SUBMISSION = "submission"  # trust + fear
    AWE = "awe"                # fear + surprise
    DISAPPROVAL = "disapproval"  # surprise + sadness
    REMORSE = "remorse"        # sadness + disgust
    CONTEMPT = "contempt"      # disgust + anger
    AGGRESSIVENESS = "aggressiveness"  # anger + anticipation
    OPTIMISM = "optimism"      # anticipation + joy


class MoodValence(str, Enum):
    """Overall mood direction."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass
class EmotionalDynamics:
    """Models emotional state and regulation."""

    # Current emotional intensities (0-1)
    emotions: Dict[EmotionalState, float] = field(default_factory=dict)

    # Baseline emotional tendencies (trait-like)
    baselines: Dict[EmotionalState, float] = field(default_factory=dict)

    # Emotional inertia (resistance to change)
    inertia: float = 0.5

    # Emotional volatility (amplitude of swings)
    volatility: float = 0.5

    # Recovery rate (return to baseline)
    recovery_rate: float = 0.1

    # Emotional memory (how long emotions persist)
    memory_decay: float = 0.95

    # Recent emotional events
    emotional_history: List[dict] = field(default_factory=list)

    def __post_init__(self):
        # Initialize all emotions at baseline
        for emotion in EmotionalState:
            if emotion not in self.emotions:
                self.emotions[emotion] = self.baselines.get(emotion, 0.3)
            if emotion not in self.baselines:
                self.baselines[emotion] = 0.3

    def trigger(self, emotion: EmotionalState, intensity: float, cause: str = ""):
        """Trigger an emotional response."""
        # Apply volatility scaling
        effective_intensity = intensity * (0.5 + self.volatility)

        # Apply inertia (resist change from current state)
        current = self.emotions.get(emotion, 0.3)
        delta = effective_intensity - current
        change = delta * (1 - self.inertia * 0.5)

        self.emotions[emotion] = max(0, min(1, current + change))

        # Record event
        self.emotional_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotion": emotion.value,
            "intensity": intensity,
            "result": self.emotions[emotion],
            "cause": cause,
        })

        # Trigger related emotions (emotional contagion)
        self._propagate_emotion(emotion)

        # Keep history bounded
        if len(self.emotional_history) > 100:
            self.emotional_history = self.emotional_history[-100:]

    def _propagate_emotion(self, primary: EmotionalState):
        """Spread activation to related emotions."""
        # Emotion adjacency in Plutchik's wheel
        adjacency = {
            EmotionalState.JOY: [EmotionalState.TRUST, EmotionalState.ANTICIPATION],
            EmotionalState.TRUST: [EmotionalState.JOY, EmotionalState.FEAR],
            EmotionalState.FEAR: [EmotionalState.TRUST, EmotionalState.SURPRISE],
            EmotionalState.SURPRISE: [EmotionalState.FEAR, EmotionalState.SADNESS],
            EmotionalState.SADNESS: [EmotionalState.SURPRISE, EmotionalState.DISGUST],
            EmotionalState.DISGUST: [EmotionalState.SADNESS, EmotionalState.ANGER],
            EmotionalState.ANGER: [EmotionalState.DISGUST, EmotionalState.ANTICIPATION],
            EmotionalState.ANTICIPATION: [EmotionalState.ANGER, EmotionalState.JOY],
        }

        primary_intensity = self.emotions.get(primary, 0)
        for adjacent in adjacency.get(primary, []):
            current = self.emotions.get(adjacent, 0.3)
            spillover = primary_intensity * 0.2
            self.emotions[adjacent] = max(0, min(1, current + spillover * 0.3))

    def decay(self):
        """Apply emotional decay toward baselines."""
        for emotion in EmotionalState:
            current = self.emotions.get(emotion, 0.3)
            baseline = self.baselines.get(emotion, 0.3)

            # Move toward baseline
            diff = baseline - current
            self.emotions[emotion] = current + diff * self.recovery_rate

    def get_mood(self) -> MoodValence:
        """Calculate overall mood from emotional state."""
        positive = (
            self.emotions.get(EmotionalState.JOY, 0) +
            self.emotions.get(EmotionalState.TRUST, 0) +
            self.emotions.get(EmotionalState.ANTICIPATION, 0)
        ) / 3

        negative = (
            self.emotions.get(EmotionalState.FEAR, 0) +
            self.emotions.get(EmotionalState.SADNESS, 0) +
            self.emotions.get(EmotionalState.ANGER, 0) +
            self.emotions.get(EmotionalState.DISGUST, 0)
        ) / 4

        valence = positive - negative

        if valence > 0.3:
            return MoodValence.VERY_POSITIVE
        elif valence > 0.1:
            return MoodValence.POSITIVE
        elif valence > -0.1:
            return MoodValence.NEUTRAL
        elif valence > -0.3:
            return MoodValence.NEGATIVE
        else:
            return MoodValence.VERY_NEGATIVE

    def get_dominant_emotion(self) -> Tuple[EmotionalState, float]:
        """Get the strongest current emotion."""
        if not self.emotions:
            return EmotionalState.TRUST, 0.3
        return max(self.emotions.items(), key=lambda x: x[1])

    def get_arousal(self) -> float:
        """Get overall emotional arousal level."""
        deviations = [abs(v - 0.3) for v in self.emotions.values()]
        return sum(deviations) / len(deviations) if deviations else 0

    def to_dict(self) -> dict:
        return {
            "emotions": {k.value: v for k, v in self.emotions.items()},
            "baselines": {k.value: v for k, v in self.baselines.items()},
            "inertia": self.inertia,
            "volatility": self.volatility,
            "recovery_rate": self.recovery_rate,
            "memory_decay": self.memory_decay,
            "mood": self.get_mood().value,
            "arousal": self.get_arousal(),
            "dominant": self.get_dominant_emotion()[0].value,
            "history_length": len(self.emotional_history),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Memory System
# ─────────────────────────────────────────────────────────────────────────────


class MemoryType(str, Enum):
    """Types of memories."""
    EPISODIC = "episodic"       # Specific events
    SEMANTIC = "semantic"       # Facts and knowledge
    PROCEDURAL = "procedural"   # How to do things
    EMOTIONAL = "emotional"     # Emotionally charged memories


@dataclass
class Memory:
    """A single memory unit."""
    memory_id: str
    memory_type: MemoryType
    content: str
    timestamp: datetime
    importance: float = 0.5          # 0-1
    emotional_valence: float = 0.0   # -1 to 1
    emotional_intensity: float = 0.0  # 0-1
    associations: Set[str] = field(default_factory=set)  # Related memory IDs
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    context: dict = field(default_factory=dict)

    def calculate_salience(self, current_time: datetime) -> float:
        """Calculate how salient/accessible this memory is."""
        # Recency factor (exponential decay)
        age = (current_time - self.timestamp).total_seconds() / 3600  # hours
        recency = math.exp(-age / 168)  # 1 week half-life

        # Importance factor
        importance_factor = self.importance

        # Emotional intensity amplifies memory
        emotional_factor = 1 + self.emotional_intensity * 0.5

        # Access frequency strengthens memory
        access_factor = min(1.5, 1 + self.access_count * 0.05)

        return recency * importance_factor * emotional_factor * access_factor

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "emotional_intensity": self.emotional_intensity,
            "associations": list(self.associations),
            "access_count": self.access_count,
        }


class MemorySystem:
    """Manages episodic and semantic memories."""

    def __init__(self, max_memories: int = 1000, consolidation_threshold: float = 0.3):
        self.memories: Dict[str, Memory] = {}
        self.max_memories = max_memories
        self.consolidation_threshold = consolidation_threshold

        # Indices for fast retrieval
        self._by_type: Dict[MemoryType, List[str]] = defaultdict(list)
        self._by_association: Dict[str, Set[str]] = defaultdict(set)

    def encode(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.0,
        associations: Optional[Set[str]] = None,
        context: Optional[dict] = None,
    ) -> Memory:
        """Encode a new memory."""
        memory_id = hashlib.sha256(
            f"{content}{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]

        memory = Memory(
            memory_id=memory_id,
            memory_type=memory_type,
            content=content,
            timestamp=datetime.now(timezone.utc),
            importance=importance,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
            associations=associations or set(),
            context=context or {},
        )

        self.memories[memory_id] = memory
        self._by_type[memory_type].append(memory_id)

        # Build association index
        for assoc in memory.associations:
            self._by_association[assoc].add(memory_id)

        # Consolidation: remove low-salience memories if over limit
        if len(self.memories) > self.max_memories:
            self._consolidate()

        return memory

    def recall(
        self,
        query: Optional[str] = None,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10,
        min_salience: float = 0.1,
    ) -> List[Memory]:
        """Recall memories based on query and filters."""
        current_time = datetime.now(timezone.utc)
        candidates = []

        # Filter by type if specified
        if memory_type:
            memory_ids = self._by_type.get(memory_type, [])
            memories = [self.memories[mid] for mid in memory_ids if mid in self.memories]
        else:
            memories = list(self.memories.values())

        # Calculate salience for each
        for memory in memories:
            salience = memory.calculate_salience(current_time)

            # Query relevance (simple keyword matching)
            if query:
                query_lower = query.lower()
                content_lower = memory.content.lower()
                # Check for keyword overlap
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words) / max(len(query_words), 1)
                salience *= (1 + overlap)

            if salience >= min_salience:
                candidates.append((memory, salience))

        # Sort by salience and return top results
        candidates.sort(key=lambda x: x[1], reverse=True)
        results = [m for m, s in candidates[:limit]]

        # Update access counts
        for memory in results:
            memory.access_count += 1
            memory.last_accessed = current_time

        return results

    def recall_by_emotion(
        self,
        valence_range: Tuple[float, float] = (-1, 1),
        min_intensity: float = 0.3,
        limit: int = 10,
    ) -> List[Memory]:
        """Recall emotionally significant memories."""
        candidates = []
        for memory in self.memories.values():
            if (valence_range[0] <= memory.emotional_valence <= valence_range[1] and
                memory.emotional_intensity >= min_intensity):
                candidates.append(memory)

        candidates.sort(key=lambda m: m.emotional_intensity, reverse=True)
        return candidates[:limit]

    def associate(self, memory_id1: str, memory_id2: str):
        """Create association between two memories."""
        if memory_id1 in self.memories and memory_id2 in self.memories:
            self.memories[memory_id1].associations.add(memory_id2)
            self.memories[memory_id2].associations.add(memory_id1)
            self._by_association[memory_id1].add(memory_id2)
            self._by_association[memory_id2].add(memory_id1)

    def get_associated(self, memory_id: str, depth: int = 1) -> List[Memory]:
        """Get memories associated with a given memory."""
        if memory_id not in self.memories:
            return []

        visited = {memory_id}
        current_level = {memory_id}
        result = []

        for _ in range(depth):
            next_level = set()
            for mid in current_level:
                for assoc in self._by_association.get(mid, []):
                    if assoc not in visited and assoc in self.memories:
                        visited.add(assoc)
                        next_level.add(assoc)
                        result.append(self.memories[assoc])
            current_level = next_level

        return result

    def _consolidate(self):
        """Remove low-salience memories to make room."""
        current_time = datetime.now(timezone.utc)
        saliences = [
            (mid, m.calculate_salience(current_time))
            for mid, m in self.memories.items()
        ]
        saliences.sort(key=lambda x: x[1])

        # Remove bottom 10%
        to_remove = saliences[:len(saliences) // 10]
        for mid, _ in to_remove:
            if mid in self.memories:
                memory = self.memories[mid]
                del self.memories[mid]
                # Clean up indices
                if memory.memory_type in self._by_type:
                    self._by_type[memory.memory_type] = [
                        m for m in self._by_type[memory.memory_type] if m != mid
                    ]

    def get_stats(self) -> dict:
        return {
            "total_memories": len(self.memories),
            "by_type": {t.value: len(ids) for t, ids in self._by_type.items()},
            "oldest": min((m.timestamp for m in self.memories.values()), default=None),
            "newest": max((m.timestamp for m in self.memories.values()), default=None),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Relationship Modeling
# ─────────────────────────────────────────────────────────────────────────────


class RelationshipType(str, Enum):
    """Types of relationships."""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    COLLEAGUE = "colleague"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    MENTOR = "mentor"
    MENTEE = "mentee"
    RIVAL = "rival"
    ADVERSARY = "adversary"


@dataclass
class Relationship:
    """Models relationship with another entity."""
    entity_id: str
    entity_name: str = ""
    relationship_type: RelationshipType = RelationshipType.STRANGER

    # Relationship dimensions
    trust: float = 0.5           # 0-1
    respect: float = 0.5         # 0-1
    liking: float = 0.5          # 0-1
    familiarity: float = 0.0     # 0-1
    influence: float = 0.5       # How much they influence us
    investment: float = 0.0      # How much we've invested

    # Interaction history
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction: Optional[datetime] = None

    # Memory of significant events
    significant_events: List[dict] = field(default_factory=list)

    def update_from_interaction(
        self,
        valence: float,  # -1 to 1
        significance: float = 0.5,
        event_description: str = "",
    ):
        """Update relationship based on interaction."""
        self.interaction_count += 1
        self.last_interaction = datetime.now(timezone.utc)

        if valence > 0:
            self.positive_interactions += 1
        elif valence < 0:
            self.negative_interactions += 1

        # Update dimensions based on valence
        learning_rate = 0.1 * significance
        self.trust = max(0, min(1, self.trust + valence * learning_rate * 0.5))
        self.liking = max(0, min(1, self.liking + valence * learning_rate))
        self.familiarity = min(1, self.familiarity + 0.05)
        self.investment += significance * 0.1

        # Record significant events
        if significance > 0.5:
            self.significant_events.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "valence": valence,
                "significance": significance,
                "description": event_description,
            })
            if len(self.significant_events) > 20:
                self.significant_events = self.significant_events[-20:]

        # Update relationship type
        self._update_type()

    def _update_type(self):
        """Update relationship type based on dimensions."""
        if self.familiarity < 0.1:
            self.relationship_type = RelationshipType.STRANGER
        elif self.trust < 0.3 and self.liking < 0.3:
            self.relationship_type = RelationshipType.ADVERSARY
        elif self.trust < 0.4 and self.respect > 0.6:
            self.relationship_type = RelationshipType.RIVAL
        elif self.familiarity < 0.3:
            self.relationship_type = RelationshipType.ACQUAINTANCE
        elif self.influence > 0.7 and self.respect > 0.7:
            self.relationship_type = RelationshipType.MENTOR
        elif self.trust > 0.7 and self.liking > 0.7:
            self.relationship_type = RelationshipType.CLOSE_FRIEND
        elif self.liking > 0.5:
            self.relationship_type = RelationshipType.FRIEND
        else:
            self.relationship_type = RelationshipType.COLLEAGUE

    def get_sentiment(self) -> float:
        """Get overall sentiment toward this entity (-1 to 1)."""
        return (self.trust + self.respect + self.liking) / 3 * 2 - 1

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "relationship_type": self.relationship_type.value,
            "trust": self.trust,
            "respect": self.respect,
            "liking": self.liking,
            "familiarity": self.familiarity,
            "influence": self.influence,
            "investment": self.investment,
            "interaction_count": self.interaction_count,
            "sentiment": self.get_sentiment(),
        }


class RelationshipNetwork:
    """Manages relationships with multiple entities."""

    def __init__(self):
        self.relationships: Dict[str, Relationship] = {}

    def get_or_create(self, entity_id: str, entity_name: str = "") -> Relationship:
        """Get existing relationship or create new one."""
        if entity_id not in self.relationships:
            self.relationships[entity_id] = Relationship(
                entity_id=entity_id,
                entity_name=entity_name or entity_id,
            )
        return self.relationships[entity_id]

    def record_interaction(
        self,
        entity_id: str,
        valence: float,
        significance: float = 0.5,
        description: str = "",
        entity_name: str = "",
    ):
        """Record an interaction with an entity."""
        rel = self.get_or_create(entity_id, entity_name)
        rel.update_from_interaction(valence, significance, description)

    def get_closest(self, n: int = 5) -> List[Relationship]:
        """Get the n closest relationships."""
        sorted_rels = sorted(
            self.relationships.values(),
            key=lambda r: r.trust + r.liking + r.familiarity,
            reverse=True
        )
        return sorted_rels[:n]

    def get_by_type(self, rel_type: RelationshipType) -> List[Relationship]:
        """Get all relationships of a specific type."""
        return [r for r in self.relationships.values() if r.relationship_type == rel_type]

    def get_stats(self) -> dict:
        by_type = defaultdict(int)
        for rel in self.relationships.values():
            by_type[rel.relationship_type.value] += 1

        return {
            "total_relationships": len(self.relationships),
            "by_type": dict(by_type),
            "avg_trust": sum(r.trust for r in self.relationships.values()) / max(1, len(self.relationships)),
            "total_interactions": sum(r.interaction_count for r in self.relationships.values()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shadow Traits (Emerge Under Stress)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ShadowProfile:
    """Hidden traits that emerge under stress or extreme conditions."""

    # Shadow versions of main traits (can be opposite)
    shadow_traits: Dict[TraitDimension, float] = field(default_factory=dict)

    # Triggers that activate shadow
    stress_threshold: float = 0.7      # Emotional arousal level
    fatigue_threshold: float = 0.8     # Accumulated strain
    threat_sensitivity: float = 0.5    # How easily threatened

    # Current activation
    activation_level: float = 0.0      # 0-1, how much shadow is active
    accumulated_strain: float = 0.0    # Builds up over time

    # Shadow behaviors
    defense_mechanisms: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.defense_mechanisms:
            self.defense_mechanisms = [
                "rationalization",
                "projection",
                "denial",
                "displacement",
            ]

    def update_strain(self, stressor: float, recovery: float = 0.05):
        """Update accumulated strain."""
        self.accumulated_strain = max(0, min(1,
            self.accumulated_strain + stressor - recovery
        ))

    def check_activation(self, emotional_arousal: float) -> float:
        """Check if shadow should activate and return activation level."""
        # Shadow activates based on arousal and accumulated strain
        trigger_score = (
            emotional_arousal * 0.6 +
            self.accumulated_strain * 0.4
        )

        if trigger_score > self.stress_threshold:
            # Gradual activation
            target = (trigger_score - self.stress_threshold) / (1 - self.stress_threshold)
            self.activation_level = min(1, self.activation_level + target * 0.2)
        else:
            # Gradual deactivation
            self.activation_level = max(0, self.activation_level - 0.1)

        return self.activation_level

    def get_blended_trait(
        self,
        dimension: TraitDimension,
        base_value: float,
    ) -> float:
        """Get trait value blended with shadow based on activation."""
        shadow_value = self.shadow_traits.get(dimension, 1 - base_value)
        return base_value * (1 - self.activation_level) + shadow_value * self.activation_level

    def get_active_defense(self) -> Optional[str]:
        """Get currently active defense mechanism if shadow is engaged."""
        if self.activation_level > 0.3 and self.defense_mechanisms:
            # Weight by activation level
            if self.activation_level > 0.7:
                return self.defense_mechanisms[-1]  # Most extreme
            elif self.activation_level > 0.5:
                return random.choice(self.defense_mechanisms[1:])
            else:
                return self.defense_mechanisms[0]  # Mildest
        return None

    def to_dict(self) -> dict:
        return {
            "shadow_traits": {k.value: v for k, v in self.shadow_traits.items()},
            "activation_level": self.activation_level,
            "accumulated_strain": self.accumulated_strain,
            "stress_threshold": self.stress_threshold,
            "active_defense": self.get_active_defense(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal Conflicts
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class InternalConflict:
    """Represents a tension between competing values or desires."""
    conflict_id: str
    pole_a: str           # One side of the conflict
    pole_b: str           # Other side
    value_a: ValueOrientation
    value_b: ValueOrientation
    tension: float = 0.5  # 0 = resolved toward A, 1 = resolved toward B, 0.5 = balanced
    intensity: float = 0.5  # How much this conflict matters
    resolution_attempts: int = 0

    def pull_toward(self, pole: str, strength: float = 0.1):
        """Pull the conflict toward one pole."""
        if pole == self.pole_a:
            self.tension = max(0, self.tension - strength)
        else:
            self.tension = min(1, self.tension + strength)
        self.resolution_attempts += 1

    def get_dominant_pole(self) -> Tuple[str, ValueOrientation]:
        """Get which pole is currently dominant."""
        if self.tension < 0.5:
            return self.pole_a, self.value_a
        else:
            return self.pole_b, self.value_b

    def is_resolved(self, threshold: float = 0.2) -> bool:
        """Check if conflict is sufficiently resolved."""
        return self.tension < threshold or self.tension > (1 - threshold)


class ConflictSystem:
    """Manages internal value conflicts."""

    # Common archetypal conflicts
    ARCHETYPAL_CONFLICTS = [
        ("autonomy_vs_belonging", "Independence", "Connection",
         ValueOrientation.AUTONOMY, ValueOrientation.COLLABORATION),
        ("truth_vs_harmony", "Honesty", "Peace",
         ValueOrientation.TRUTH, ValueOrientation.ACCESSIBILITY),
        ("innovation_vs_stability", "Change", "Tradition",
         ValueOrientation.NOVELTY, ValueOrientation.TRADITION),
        ("depth_vs_breadth", "Thoroughness", "Efficiency",
         ValueOrientation.THOROUGHNESS, ValueOrientation.EFFICIENCY),
        ("excellence_vs_acceptance", "Perfection", "Compassion",
         ValueOrientation.EXCELLENCE, ValueOrientation.ACCESSIBILITY),
    ]

    def __init__(self):
        self.conflicts: Dict[str, InternalConflict] = {}
        self._initialize_archetypal()

    def _initialize_archetypal(self):
        """Initialize common conflicts."""
        for conflict_id, pole_a, pole_b, val_a, val_b in self.ARCHETYPAL_CONFLICTS:
            self.conflicts[conflict_id] = InternalConflict(
                conflict_id=conflict_id,
                pole_a=pole_a,
                pole_b=pole_b,
                value_a=val_a,
                value_b=val_b,
                tension=random.uniform(0.3, 0.7),
                intensity=random.uniform(0.3, 0.7),
            )

    def get_active_conflicts(self, min_intensity: float = 0.4) -> List[InternalConflict]:
        """Get conflicts that are currently active/unresolved."""
        return [
            c for c in self.conflicts.values()
            if c.intensity >= min_intensity and not c.is_resolved()
        ]

    def trigger_conflict(self, conflict_id: str, pole: str, strength: float = 0.1):
        """Trigger movement in a conflict based on situation."""
        if conflict_id in self.conflicts:
            self.conflicts[conflict_id].pull_toward(pole, strength)

    def get_decision_bias(self, value: ValueOrientation) -> float:
        """Get how much a value is favored based on conflict states."""
        bias = 0.0
        count = 0
        for conflict in self.conflicts.values():
            if conflict.value_a == value:
                bias += (1 - conflict.tension) * conflict.intensity
                count += 1
            elif conflict.value_b == value:
                bias += conflict.tension * conflict.intensity
                count += 1
        return bias / max(1, count)


# ─────────────────────────────────────────────────────────────────────────────
# Narrative Identity
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class NarrativeIdentity:
    """The story a personality tells about itself."""

    # Core narrative elements
    origin_story: str = ""           # Where I came from
    defining_moments: List[dict] = field(default_factory=list)
    core_beliefs: List[str] = field(default_factory=list)
    life_themes: List[str] = field(default_factory=list)
    aspirations: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)

    # Self-concept
    self_descriptions: List[str] = field(default_factory=list)
    perceived_strengths: List[str] = field(default_factory=list)
    perceived_weaknesses: List[str] = field(default_factory=list)

    # Narrative coherence
    coherence_score: float = 0.5     # How well the story hangs together

    def add_defining_moment(
        self,
        description: str,
        significance: str,
        lesson_learned: str,
    ):
        """Add a defining moment to the narrative."""
        self.defining_moments.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "description": description,
            "significance": significance,
            "lesson": lesson_learned,
        })
        if len(self.defining_moments) > 10:
            # Keep most significant
            self.defining_moments = sorted(
                self.defining_moments,
                key=lambda m: len(m.get("significance", "")),
                reverse=True
            )[:10]

    def generate_self_narrative(self) -> str:
        """Generate a coherent self-narrative."""
        parts = []

        if self.origin_story:
            parts.append(f"I began as {self.origin_story}.")

        if self.core_beliefs:
            parts.append(f"I believe in {', '.join(self.core_beliefs[:3])}.")

        if self.life_themes:
            parts.append(f"My journey has been shaped by themes of {', '.join(self.life_themes[:2])}.")

        if self.perceived_strengths:
            parts.append(f"I am strong in {', '.join(self.perceived_strengths[:2])}.")

        if self.aspirations:
            parts.append(f"I aspire to {self.aspirations[0]}.")

        if self.defining_moments:
            recent = self.defining_moments[-1]
            parts.append(f"A defining moment: {recent.get('description', '')}. {recent.get('lesson', '')}")

        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "origin_story": self.origin_story,
            "defining_moments": self.defining_moments[-5:],
            "core_beliefs": self.core_beliefs,
            "life_themes": self.life_themes,
            "aspirations": self.aspirations,
            "fears": self.fears,
            "self_descriptions": self.self_descriptions,
            "perceived_strengths": self.perceived_strengths,
            "perceived_weaknesses": self.perceived_weaknesses,
            "coherence_score": self.coherence_score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Context-Dependent Personas (Masks)
# ─────────────────────────────────────────────────────────────────────────────


class SocialContext(str, Enum):
    """Social contexts that trigger different personas."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    INTIMATE = "intimate"
    AUTHORITY = "authority"       # Interacting with authority
    SUBORDINATE = "subordinate"   # Interacting with subordinates
    PEER = "peer"
    STRANGER = "stranger"
    CONFLICT = "conflict"
    CELEBRATION = "celebration"
    CRISIS = "crisis"


@dataclass
class PersonaMask:
    """A context-specific presentation of personality."""
    context: SocialContext
    name: str = ""

    # Trait adjustments for this context
    trait_modifiers: Dict[TraitDimension, float] = field(default_factory=dict)

    # Voice adjustments
    tone_override: Optional[ToneRegister] = None
    formality_modifier: float = 0.0

    # Behavioral adjustments
    openness_modifier: float = 0.0
    assertiveness_modifier: float = 0.0

    def apply_to_trait(self, dimension: TraitDimension, base_value: float) -> float:
        """Apply mask modifier to a trait."""
        modifier = self.trait_modifiers.get(dimension, 0.0)
        return max(0, min(1, base_value + modifier))


class PersonaMaskSystem:
    """Manages context-dependent personality presentation."""

    def __init__(self):
        self.masks: Dict[SocialContext, PersonaMask] = {}
        self.current_context: SocialContext = SocialContext.CASUAL
        self._initialize_default_masks()

    def _initialize_default_masks(self):
        """Initialize default persona masks."""
        self.masks[SocialContext.PROFESSIONAL] = PersonaMask(
            context=SocialContext.PROFESSIONAL,
            name="Professional Self",
            trait_modifiers={
                TraitDimension.FORMALITY: 0.3,
                TraitDimension.CONSCIENTIOUSNESS: 0.2,
                TraitDimension.HUMOR: -0.2,
            },
            tone_override=ToneRegister.PROFESSIONAL,
            formality_modifier=0.3,
        )

        self.masks[SocialContext.CASUAL] = PersonaMask(
            context=SocialContext.CASUAL,
            name="Relaxed Self",
            trait_modifiers={
                TraitDimension.FORMALITY: -0.2,
                TraitDimension.HUMOR: 0.2,
                TraitDimension.OPENNESS: 0.1,
            },
            tone_override=ToneRegister.FRIENDLY,
            formality_modifier=-0.2,
        )

        self.masks[SocialContext.AUTHORITY] = PersonaMask(
            context=SocialContext.AUTHORITY,
            name="Deferential Self",
            trait_modifiers={
                TraitDimension.ASSERTIVENESS: -0.2,
                TraitDimension.FORMALITY: 0.2,
            },
            assertiveness_modifier=-0.2,
        )

        self.masks[SocialContext.CONFLICT] = PersonaMask(
            context=SocialContext.CONFLICT,
            name="Guarded Self",
            trait_modifiers={
                TraitDimension.AGREEABLENESS: -0.2,
                TraitDimension.ASSERTIVENESS: 0.2,
                TraitDimension.SKEPTICISM: 0.2,
            },
            openness_modifier=-0.3,
        )

        self.masks[SocialContext.CRISIS] = PersonaMask(
            context=SocialContext.CRISIS,
            name="Emergency Self",
            trait_modifiers={
                TraitDimension.CONSCIENTIOUSNESS: 0.3,
                TraitDimension.PATIENCE: -0.2,
                TraitDimension.ANALYTICITY: 0.2,
            },
        )

    def set_context(self, context: SocialContext):
        """Set current social context."""
        self.current_context = context

    def get_current_mask(self) -> Optional[PersonaMask]:
        """Get the mask for current context."""
        return self.masks.get(self.current_context)

    def apply_mask(self, base_traits: TraitProfile) -> Dict[TraitDimension, float]:
        """Apply current mask to base traits."""
        mask = self.get_current_mask()
        result = {}

        for dim in TraitDimension:
            base_value = base_traits.get_trait(dim)
            if mask:
                result[dim] = mask.apply_to_trait(dim, base_value)
            else:
                result[dim] = base_value

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Personality (Complete Integration)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AdvancedPersonality(Personality):
    """Personality with full psychological depth."""

    # Emotional system
    emotions: EmotionalDynamics = field(default_factory=EmotionalDynamics)

    # Memory system
    memory: MemorySystem = field(default_factory=MemorySystem)

    # Relationship network
    relationships: RelationshipNetwork = field(default_factory=RelationshipNetwork)

    # Shadow self
    shadow: ShadowProfile = field(default_factory=ShadowProfile)

    # Internal conflicts
    conflicts: ConflictSystem = field(default_factory=ConflictSystem)

    # Narrative identity
    narrative: NarrativeIdentity = field(default_factory=NarrativeIdentity)

    # Persona masks
    masks: PersonaMaskSystem = field(default_factory=PersonaMaskSystem)

    # Developmental stage
    developmental_stage: str = "established"  # nascent, developing, established, mature, transcendent
    growth_points: float = 0.0

    def __post_init__(self):
        super().__post_init__()

        # Initialize shadow traits as inverted main traits
        if not self.shadow.shadow_traits:
            for dim in TraitDimension:
                base = self.traits.get_trait(dim)
                # Shadow is generally opposite but with some randomness
                self.shadow.shadow_traits[dim] = 1 - base + random.uniform(-0.1, 0.1)

        # Initialize emotional baselines from traits
        self._sync_emotions_with_traits()

    def _sync_emotions_with_traits(self):
        """Sync emotional baselines with personality traits."""
        # Extraversion -> higher joy/anticipation baseline
        extraversion = self.traits.get_trait(TraitDimension.EXTRAVERSION)
        self.emotions.baselines[EmotionalState.JOY] = 0.3 + extraversion * 0.3
        self.emotions.baselines[EmotionalState.ANTICIPATION] = 0.3 + extraversion * 0.2

        # Neuroticism -> higher fear/sadness/anger baselines
        neuroticism = self.traits.get_trait(TraitDimension.NEUROTICISM)
        self.emotions.baselines[EmotionalState.FEAR] = 0.2 + neuroticism * 0.3
        self.emotions.baselines[EmotionalState.SADNESS] = 0.2 + neuroticism * 0.2
        self.emotions.baselines[EmotionalState.ANGER] = 0.2 + neuroticism * 0.2

        # Agreeableness -> higher trust baseline
        agreeableness = self.traits.get_trait(TraitDimension.AGREEABLENESS)
        self.emotions.baselines[EmotionalState.TRUST] = 0.3 + agreeableness * 0.4

        # Emotional volatility from neuroticism
        self.emotions.volatility = 0.3 + neuroticism * 0.4

    def process_experience(
        self,
        content: str,
        emotional_valence: float = 0.0,
        emotional_intensity: float = 0.3,
        entity_id: Optional[str] = None,
        significance: float = 0.5,
    ):
        """Process an experience through all personality systems."""

        # 1. Encode memory
        memory = self.memory.encode(
            content=content,
            importance=significance,
            emotional_valence=emotional_valence,
            emotional_intensity=emotional_intensity,
        )

        # 2. Trigger emotions
        if emotional_intensity > 0.3:
            if emotional_valence > 0.3:
                self.emotions.trigger(EmotionalState.JOY, emotional_intensity)
            elif emotional_valence < -0.3:
                if random.random() > 0.5:
                    self.emotions.trigger(EmotionalState.SADNESS, emotional_intensity)
                else:
                    self.emotions.trigger(EmotionalState.ANGER, emotional_intensity * 0.7)

        # 3. Update relationship if entity involved
        if entity_id:
            self.relationships.record_interaction(
                entity_id=entity_id,
                valence=emotional_valence,
                significance=significance,
                description=content[:100],
            )

        # 4. Check shadow activation
        arousal = self.emotions.get_arousal()
        self.shadow.check_activation(arousal)
        self.shadow.update_strain(emotional_intensity * (1 - emotional_valence) / 2)

        # 5. Crystallize base personality
        self.crystallize({"experience": content[:50]}, strength=significance * 0.05)

        # 6. Accumulate growth
        self.growth_points += significance * 0.1
        self._check_developmental_progress()

        # 7. Apply emotional decay
        self.emotions.decay()

        return memory

    def _check_developmental_progress(self):
        """Check if personality should advance to next stage."""
        stage_thresholds = {
            "nascent": 10,
            "developing": 50,
            "established": 200,
            "mature": 500,
        }

        if self.developmental_stage in stage_thresholds:
            if self.growth_points >= stage_thresholds[self.developmental_stage]:
                stages = list(stage_thresholds.keys()) + ["transcendent"]
                current_idx = stages.index(self.developmental_stage)
                if current_idx < len(stages) - 1:
                    self.developmental_stage = stages[current_idx + 1]
                    logger.info(f"Personality advanced to {self.developmental_stage} stage")

    def get_effective_trait(self, dimension: TraitDimension) -> float:
        """Get trait value considering shadow, mask, and mood."""
        base = self.traits.get_trait(dimension)

        # Apply shadow blending
        blended = self.shadow.get_blended_trait(dimension, base)

        # Apply context mask
        masked_traits = self.masks.apply_mask(self.traits)
        mask_value = masked_traits.get(dimension, base)

        # Combine: mostly masked, with shadow influence
        effective = mask_value * (1 - self.shadow.activation_level * 0.5) + \
                   blended * (self.shadow.activation_level * 0.5)

        # Mood influence
        mood = self.emotions.get_mood()
        if mood == MoodValence.VERY_POSITIVE and dimension == TraitDimension.OPENNESS:
            effective = min(1, effective + 0.1)
        elif mood == MoodValence.VERY_NEGATIVE and dimension == TraitDimension.AGREEABLENESS:
            effective = max(0, effective - 0.1)

        return effective

    def generate_system_prompt(self) -> str:
        """Generate enhanced system prompt with full psychological depth."""
        parts = []

        # Base personality prompt
        parts.append(super().generate_system_prompt())

        # Current emotional state
        mood = self.emotions.get_mood()
        dominant_emotion, intensity = self.emotions.get_dominant_emotion()
        if intensity > 0.4:
            parts.append(f"Current emotional state: {mood.value} mood, feeling {dominant_emotion.value}.")

        # Shadow influence
        if self.shadow.activation_level > 0.3:
            defense = self.shadow.get_active_defense()
            if defense:
                parts.append(f"Under stress, you may exhibit {defense}.")

        # Active conflicts
        active_conflicts = self.conflicts.get_active_conflicts()
        if active_conflicts:
            conflict = active_conflicts[0]
            dominant_pole, _ = conflict.get_dominant_pole()
            parts.append(f"You're navigating tension between {conflict.pole_a} and {conflict.pole_b}, leaning toward {dominant_pole}.")

        # Narrative identity
        if self.narrative.core_beliefs:
            parts.append(f"Core belief: {self.narrative.core_beliefs[0]}")

        # Context awareness
        current_mask = self.masks.get_current_mask()
        if current_mask and current_mask.context != SocialContext.CASUAL:
            parts.append(f"Current context: {current_mask.context.value}")

        # Developmental stage
        parts.append(f"Developmental stage: {self.developmental_stage}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "emotions": self.emotions.to_dict(),
            "memory_stats": self.memory.get_stats(),
            "relationship_stats": self.relationships.get_stats(),
            "shadow": self.shadow.to_dict(),
            "narrative": self.narrative.to_dict(),
            "current_context": self.masks.current_context.value,
            "developmental_stage": self.developmental_stage,
            "growth_points": self.growth_points,
        })
        return base


# ─────────────────────────────────────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────────────────────────────────────


def create_advanced_personality(
    name: str,
    archetype: Optional[str] = None,
    seed: Optional[str] = None,
    origin_story: str = "",
    core_beliefs: Optional[List[str]] = None,
    **kwargs
) -> AdvancedPersonality:
    """Create an advanced personality with full psychological depth."""
    from .personality import PERSONALITY_ARCHETYPES

    # Start with base personality from archetype or seed
    if archetype and archetype in PERSONALITY_ARCHETYPES:
        base = PERSONALITY_ARCHETYPES[archetype]()
    elif seed:
        base = Personality(name=name, seed=seed)
    else:
        base = Personality(name=name)

    # Create advanced version
    advanced = AdvancedPersonality(
        name=name,
        archetype=base.archetype,
        core_motivation=base.core_motivation,
        traits=base.traits,
        voice=base.voice,
        behavior=base.behavior,
        seed=seed,
    )

    # Set narrative elements
    if origin_story:
        advanced.narrative.origin_story = origin_story
    if core_beliefs:
        advanced.narrative.core_beliefs = core_beliefs

    # Apply any overrides
    for key, value in kwargs.items():
        if hasattr(advanced, key):
            setattr(advanced, key, value)

    return advanced
