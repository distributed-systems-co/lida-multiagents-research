"""
Advanced Intelligence Analytics System.

Provides:
- Actor detection, tracking, and profiling
- Actor network/coalition analysis
- Pattern detection algorithms
- Pluggable analytics architecture
- Anomaly detection
- Trend analysis
"""

from __future__ import annotations
import asyncio
import math
import re
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Set, Optional, Tuple, Any, Callable,
    Protocol, Type, TypeVar, Generic, Union
)
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ACTOR SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ActorType(str, Enum):
    """Types of actors in the international system."""
    GOVERNMENT = "government"
    MILITARY = "military"
    REBEL = "rebel"
    OPPOSITION = "opposition"
    POLITICAL_PARTY = "political_party"
    ETHNIC_GROUP = "ethnic_group"
    RELIGIOUS_GROUP = "religious_group"
    CRIMINAL = "criminal"
    TERRORIST = "terrorist"
    MEDIA = "media"
    EDUCATION = "education"
    BUSINESS = "business"
    NGO = "ngo"
    IGO = "igo"
    REFUGEE = "refugee"
    CIVILIAN = "civilian"
    HEALTH = "health"
    LABOR = "labor"
    UNKNOWN = "unknown"


class ActorAffiliation(str, Enum):
    """Actor alignment/affiliation."""
    STATE = "state"           # Government-aligned
    PRO_STATE = "pro_state"   # Supports current government
    ANTI_STATE = "anti_state" # Opposition to government
    NON_STATE = "non_state"   # Independent actors
    TRANSNATIONAL = "transnational"  # Cross-border actors


@dataclass
class ActorProfile:
    """Comprehensive actor profile."""
    actor_id: str
    code: str
    name: str

    # Classification
    actor_type: ActorType = ActorType.UNKNOWN
    affiliation: ActorAffiliation = ActorAffiliation.NON_STATE

    # Country association
    country_code: str = ""
    country_name: str = ""

    # Derived attributes
    alternative_names: Set[str] = field(default_factory=set)
    known_aliases: Set[str] = field(default_factory=set)

    # Activity statistics
    total_events: int = 0
    events_as_actor1: int = 0
    events_as_actor2: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    # Behavioral metrics
    aggression_score: float = 0.0       # 0-1, how often initiates conflict
    cooperation_score: float = 0.0      # 0-1, how often cooperates
    activity_level: float = 0.0         # events per day
    influence_score: float = 0.0        # based on media mentions
    volatility_score: float = 0.0       # variance in behavior

    # Event breakdown
    event_types: Dict[str, int] = field(default_factory=dict)
    quad_distribution: Dict[int, int] = field(default_factory=dict)

    # Relationships
    allies: List[Tuple[str, float]] = field(default_factory=list)      # (actor_id, strength)
    adversaries: List[Tuple[str, float]] = field(default_factory=list)

    # Goldstein scores
    avg_goldstein: float = 0.0
    min_goldstein: float = 0.0
    max_goldstein: float = 0.0

    def get_behavior_signature(self) -> str:
        """Get a behavioral signature for pattern matching."""
        return f"{self.actor_type.value}:{self.affiliation.value}:A{self.aggression_score:.2f}:C{self.cooperation_score:.2f}"


@dataclass
class ActorRelationship:
    """Relationship between two actors."""
    actor1_id: str
    actor2_id: str

    # Interaction counts
    total_interactions: int = 0
    cooperative_interactions: int = 0
    conflictual_interactions: int = 0

    # Sentiment
    avg_tone: float = 0.0
    avg_goldstein: float = 0.0

    # Trend
    trend: str = "stable"  # improving, stable, deteriorating
    volatility: float = 0.0

    # Time
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None

    @property
    def relationship_type(self) -> str:
        if self.total_interactions == 0:
            return "unknown"
        coop_ratio = self.cooperative_interactions / self.total_interactions
        if coop_ratio > 0.7:
            return "allied"
        elif coop_ratio > 0.5:
            return "friendly"
        elif coop_ratio > 0.3:
            return "neutral"
        elif coop_ratio > 0.15:
            return "tense"
        else:
            return "hostile"


class ActorRegistry:
    """Registry for tracking and managing actors."""

    # CAMEO actor type codes
    ACTOR_TYPE_CODES = {
        "GOV": ActorType.GOVERNMENT,
        "MIL": ActorType.MILITARY,
        "REB": ActorType.REBEL,
        "OPP": ActorType.OPPOSITION,
        "PTY": ActorType.POLITICAL_PARTY,
        "ETH": ActorType.ETHNIC_GROUP,
        "REL": ActorType.RELIGIOUS_GROUP,
        "CRM": ActorType.CRIMINAL,
        "CVL": ActorType.CIVILIAN,
        "MED": ActorType.MEDIA,
        "EDU": ActorType.EDUCATION,
        "BUS": ActorType.BUSINESS,
        "NGO": ActorType.NGO,
        "IGO": ActorType.IGO,
        "REF": ActorType.REFUGEE,
        "HLH": ActorType.HEALTH,
        "LAB": ActorType.LABOR,
    }

    def __init__(self):
        self.actors: Dict[str, ActorProfile] = {}
        self.relationships: Dict[str, ActorRelationship] = {}
        self.code_to_id: Dict[str, str] = {}
        self.name_to_id: Dict[str, str] = {}

    def _generate_actor_id(self, code: str, name: str) -> str:
        """Generate consistent actor ID."""
        key = f"{code}:{name}".lower()
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _parse_actor_type(self, type_code: str) -> ActorType:
        """Parse CAMEO actor type code."""
        if not type_code:
            return ActorType.UNKNOWN
        for prefix, atype in self.ACTOR_TYPE_CODES.items():
            if type_code.startswith(prefix):
                return atype
        return ActorType.UNKNOWN

    def get_or_create_actor(self, code: str, name: str, country: str = "",
                            type_code: str = "") -> ActorProfile:
        """Get existing actor or create new one."""
        actor_id = self._generate_actor_id(code, name)

        if actor_id not in self.actors:
            self.actors[actor_id] = ActorProfile(
                actor_id=actor_id,
                code=code,
                name=name,
                country_code=country,
                actor_type=self._parse_actor_type(type_code),
            )
            if code:
                self.code_to_id[code.lower()] = actor_id
            if name:
                self.name_to_id[name.lower()] = actor_id

        return self.actors[actor_id]

    def get_actor_by_code(self, code: str) -> Optional[ActorProfile]:
        """Lookup actor by code."""
        actor_id = self.code_to_id.get(code.lower())
        return self.actors.get(actor_id) if actor_id else None

    def get_actor_by_name(self, name: str) -> Optional[ActorProfile]:
        """Lookup actor by name."""
        actor_id = self.name_to_id.get(name.lower())
        return self.actors.get(actor_id) if actor_id else None

    def get_or_create_relationship(self, actor1_id: str, actor2_id: str) -> ActorRelationship:
        """Get or create relationship between actors."""
        # Normalize order for consistent key
        key = tuple(sorted([actor1_id, actor2_id]))
        rel_key = f"{key[0]}:{key[1]}"

        if rel_key not in self.relationships:
            self.relationships[rel_key] = ActorRelationship(
                actor1_id=key[0],
                actor2_id=key[1],
            )

        return self.relationships[rel_key]

    def get_actor_network(self, actor_id: str, depth: int = 1) -> Dict[str, Any]:
        """Get actor's relationship network."""
        visited = {actor_id}
        network = {"nodes": [], "edges": []}

        def explore(aid: str, current_depth: int):
            if current_depth > depth:
                return

            actor = self.actors.get(aid)
            if actor:
                network["nodes"].append({
                    "id": aid,
                    "name": actor.name,
                    "type": actor.actor_type.value,
                    "country": actor.country_code,
                })

            for rel_key, rel in self.relationships.items():
                if aid in (rel.actor1_id, rel.actor2_id):
                    other_id = rel.actor2_id if rel.actor1_id == aid else rel.actor1_id

                    network["edges"].append({
                        "source": rel.actor1_id,
                        "target": rel.actor2_id,
                        "type": rel.relationship_type,
                        "weight": rel.total_interactions,
                    })

                    if other_id not in visited:
                        visited.add(other_id)
                        explore(other_id, current_depth + 1)

        explore(actor_id, 0)
        return network

    def get_top_actors(self, n: int = 20, by: str = "activity") -> List[ActorProfile]:
        """Get top N actors by specified metric."""
        if by == "activity":
            return sorted(self.actors.values(), key=lambda a: -a.total_events)[:n]
        elif by == "aggression":
            return sorted(self.actors.values(), key=lambda a: -a.aggression_score)[:n]
        elif by == "influence":
            return sorted(self.actors.values(), key=lambda a: -a.influence_score)[:n]
        elif by == "volatility":
            return sorted(self.actors.values(), key=lambda a: -a.volatility_score)[:n]
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# ACTOR COALITION / GROUP DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ActorCoalition:
    """A group of actors that frequently cooperate."""
    coalition_id: str
    name: str
    members: Set[str] = field(default_factory=set)  # actor_ids

    # Characteristics
    cohesion_score: float = 0.0      # How tightly bound (0-1)
    activity_level: float = 0.0
    avg_cooperation: float = 0.0

    # Common attributes
    dominant_country: str = ""
    dominant_type: ActorType = ActorType.UNKNOWN

    # Actions
    total_events: int = 0
    cooperative_events: int = 0
    conflictual_events: int = 0

    # Targets
    common_adversaries: List[str] = field(default_factory=list)
    common_allies: List[str] = field(default_factory=list)

    def get_member_overlap(self, other: "ActorCoalition") -> float:
        """Calculate membership overlap with another coalition."""
        if not self.members or not other.members:
            return 0.0
        intersection = len(self.members & other.members)
        union = len(self.members | other.members)
        return intersection / union if union > 0 else 0.0


class CoalitionDetector:
    """Detects actor coalitions using graph community detection."""

    def __init__(self, registry: ActorRegistry):
        self.registry = registry
        self.coalitions: Dict[str, ActorCoalition] = {}

    def detect_coalitions(self, min_size: int = 2,
                          min_cooperation: float = 0.6) -> List[ActorCoalition]:
        """
        Detect coalitions using cooperation patterns.

        Uses a simplified community detection based on:
        - Cooperative interaction frequency
        - Shared adversaries
        - Actor type similarity
        """
        # Build cooperation graph
        coop_graph: Dict[str, Dict[str, float]] = defaultdict(dict)

        for rel in self.registry.relationships.values():
            if rel.total_interactions >= 3:  # Minimum interactions
                coop_ratio = rel.cooperative_interactions / rel.total_interactions
                if coop_ratio >= min_cooperation:
                    coop_graph[rel.actor1_id][rel.actor2_id] = coop_ratio
                    coop_graph[rel.actor2_id][rel.actor1_id] = coop_ratio

        # Find connected components with high cooperation
        visited = set()
        coalitions = []

        for actor_id in coop_graph:
            if actor_id in visited:
                continue

            # BFS to find coalition members
            coalition_members = set()
            queue = [actor_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                coalition_members.add(current)

                for neighbor, weight in coop_graph.get(current, {}).items():
                    if neighbor not in visited and weight >= min_cooperation:
                        queue.append(neighbor)

            if len(coalition_members) >= min_size:
                coalition = self._build_coalition(coalition_members)
                coalitions.append(coalition)

        # Sort by size and cohesion
        coalitions.sort(key=lambda c: (-len(c.members), -c.cohesion_score))

        for i, c in enumerate(coalitions):
            c.coalition_id = f"COAL_{i:03d}"
            c.name = self._generate_coalition_name(c)
            self.coalitions[c.coalition_id] = c

        return coalitions

    def _build_coalition(self, members: Set[str]) -> ActorCoalition:
        """Build coalition profile from member set."""
        coalition = ActorCoalition(
            coalition_id="",
            name="",
            members=members,
        )

        # Calculate cohesion (avg cooperation between members)
        coop_scores = []
        for m1 in members:
            for m2 in members:
                if m1 < m2:
                    key = f"{m1}:{m2}"
                    rel = self.registry.relationships.get(key)
                    if rel and rel.total_interactions > 0:
                        coop_scores.append(rel.cooperative_interactions / rel.total_interactions)

        coalition.cohesion_score = sum(coop_scores) / len(coop_scores) if coop_scores else 0

        # Find dominant country and type
        country_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for mid in members:
            actor = self.registry.actors.get(mid)
            if actor:
                coalition.total_events += actor.total_events
                country_counts[actor.country_code] += 1
                type_counts[actor.actor_type] += 1

        if country_counts:
            coalition.dominant_country = max(country_counts, key=country_counts.get)
        if type_counts:
            coalition.dominant_type = max(type_counts, key=type_counts.get)

        return coalition

    def _generate_coalition_name(self, coalition: ActorCoalition) -> str:
        """Generate descriptive name for coalition."""
        if coalition.dominant_country:
            return f"{coalition.dominant_country} {coalition.dominant_type.value} coalition"
        return f"Coalition ({len(coalition.members)} members)"


# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class PatternType(str, Enum):
    """Types of patterns that can be detected."""
    ESCALATION = "escalation"
    DE_ESCALATION = "de_escalation"
    CYCLE = "cycle"                     # Recurring patterns
    SURGE = "surge"                     # Sudden increase in activity
    LULL = "lull"                       # Sudden decrease
    SPILLOVER = "spillover"             # Conflict spreading
    CONTAGION = "contagion"             # Behavior spreading
    ANOMALY = "anomaly"                 # Unusual activity
    COORDINATION = "coordination"       # Synchronized actions
    RETALIATION = "retaliation"         # Tit-for-tat
    ALLIANCE_FORMATION = "alliance"     # New cooperation
    ALLIANCE_DISSOLUTION = "dissolution"


@dataclass
class DetectedPattern:
    """A detected pattern in intelligence data."""
    pattern_id: str
    pattern_type: PatternType
    confidence: float                   # 0-1

    # Temporal
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # Actors involved
    actors: List[str] = field(default_factory=list)
    countries: List[str] = field(default_factory=list)

    # Details
    description: str = ""
    evidence: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Severity
    severity: float = 0.0  # 0-1

    def to_alert(self) -> str:
        """Generate alert text."""
        severity_indicator = "🔴" if self.severity > 0.7 else "🟠" if self.severity > 0.4 else "🟡"
        return (f"{severity_indicator} [{self.pattern_type.value.upper()}] "
                f"Confidence: {self.confidence:.0%}\n"
                f"   {self.description}\n"
                f"   Actors: {', '.join(self.actors[:5])}")


class PatternDetector(ABC):
    """Abstract base class for pattern detectors."""

    @property
    @abstractmethod
    def pattern_type(self) -> PatternType:
        """Type of pattern this detector finds."""
        pass

    @abstractmethod
    def detect(self, events: List[Any],
               time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        """
        Detect patterns in event stream.

        Args:
            events: List of IntelEvent objects
            time_window: Time window for pattern detection

        Returns:
            List of detected patterns
        """
        pass


class EscalationDetector(PatternDetector):
    """Detects escalation patterns between actors."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ESCALATION

    def detect(self, events: List[Any],
               time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        patterns = []

        # Group events by actor pair
        pair_events: Dict[str, List[Any]] = defaultdict(list)
        for e in events:
            if e.actor1_country and e.actor2_country:
                key = tuple(sorted([e.actor1_country, e.actor2_country]))
                pair_events[str(key)].append(e)

        for pair_key, pair_events_list in pair_events.items():
            if len(pair_events_list) < 5:
                continue

            # Sort by time
            pair_events_list.sort(key=lambda x: x.timestamp)

            # Look for escalation: decreasing goldstein scores over time
            goldsteins = [e.goldstein_scale for e in pair_events_list]

            # Calculate trend
            n = len(goldsteins)
            if n < 3:
                continue

            # Simple linear regression
            x_mean = (n - 1) / 2
            y_mean = sum(goldsteins) / n

            numerator = sum((i - x_mean) * (g - y_mean) for i, g in enumerate(goldsteins))
            denominator = sum((i - x_mean) ** 2 for i in range(n))

            slope = numerator / denominator if denominator != 0 else 0

            # Significant negative slope = escalation
            if slope < -0.3:
                # Get actors
                actors = set()
                countries = set()
                for e in pair_events_list:
                    actors.add(e.actor1_name)
                    actors.add(e.actor2_name)
                    countries.add(e.actor1_country)
                    countries.add(e.actor2_country)

                pattern = DetectedPattern(
                    pattern_id=f"ESC_{pair_key}_{datetime.now().strftime('%Y%m%d%H%M')}",
                    pattern_type=PatternType.ESCALATION,
                    confidence=min(1.0, abs(slope) / 2),
                    start_time=pair_events_list[0].timestamp,
                    end_time=pair_events_list[-1].timestamp,
                    actors=list(actors)[:10],
                    countries=list(countries),
                    description=f"Escalation detected between {', '.join(countries)}",
                    severity=min(1.0, abs(slope)),
                    metrics={
                        "slope": slope,
                        "start_goldstein": goldsteins[0],
                        "end_goldstein": goldsteins[-1],
                        "event_count": n,
                    }
                )
                patterns.append(pattern)

        return patterns


class SurgeDetector(PatternDetector):
    """Detects sudden surges in activity."""

    def __init__(self, threshold_multiplier: float = 2.0):
        self.threshold_multiplier = threshold_multiplier

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.SURGE

    def detect(self, events: List[Any],
               time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        patterns = []

        # Group events by hour and country
        hourly_country_events: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for e in events:
            hour_key = e.timestamp.strftime("%Y%m%d%H")
            country = e.actor1_country or e.actor2_country or "UNKNOWN"
            hourly_country_events[hour_key][country] += 1

        # Calculate baselines per country
        country_hourly_avg: Dict[str, float] = defaultdict(float)
        country_hourly_std: Dict[str, float] = defaultdict(float)
        country_all_hours: Dict[str, List[int]] = defaultdict(list)

        for hour_data in hourly_country_events.values():
            for country, count in hour_data.items():
                country_all_hours[country].append(count)

        for country, counts in country_all_hours.items():
            if len(counts) >= 3:
                country_hourly_avg[country] = sum(counts) / len(counts)
                variance = sum((c - country_hourly_avg[country]) ** 2 for c in counts) / len(counts)
                country_hourly_std[country] = math.sqrt(variance)

        # Detect surges
        sorted_hours = sorted(hourly_country_events.keys())

        for hour in sorted_hours[-24:]:  # Last 24 hours
            for country, count in hourly_country_events[hour].items():
                avg = country_hourly_avg.get(country, 0)
                std = country_hourly_std.get(country, 1)

                if avg > 0 and std > 0:
                    z_score = (count - avg) / std

                    if z_score >= self.threshold_multiplier:
                        pattern = DetectedPattern(
                            pattern_id=f"SURGE_{country}_{hour}",
                            pattern_type=PatternType.SURGE,
                            confidence=min(1.0, z_score / 5),
                            start_time=datetime.strptime(hour, "%Y%m%d%H").replace(tzinfo=timezone.utc),
                            countries=[country],
                            description=f"Activity surge in {country}: {count} events (avg: {avg:.1f})",
                            severity=min(1.0, z_score / 4),
                            metrics={
                                "event_count": count,
                                "avg_count": avg,
                                "z_score": z_score,
                            }
                        )
                        patterns.append(pattern)

        return patterns


class RetaliationDetector(PatternDetector):
    """Detects tit-for-tat retaliation patterns."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.RETALIATION

    def detect(self, events: List[Any],
               time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        patterns = []

        # Look for A attacks B, then B attacks A
        conflict_events = [e for e in events if e.quad_class in (3, 4)]

        # Group by actor pair with direction
        directed_events: Dict[str, List[Any]] = defaultdict(list)

        for e in conflict_events:
            if e.actor1_country and e.actor2_country:
                key = f"{e.actor1_country}->{e.actor2_country}"
                directed_events[key].append(e)

        # Look for reverse patterns
        for key, evts in directed_events.items():
            parts = key.split("->")
            if len(parts) != 2:
                continue

            reverse_key = f"{parts[1]}->{parts[0]}"
            reverse_evts = directed_events.get(reverse_key, [])

            if not reverse_evts:
                continue

            # Check for temporal sequence
            for e1 in evts:
                for e2 in reverse_evts:
                    time_diff = (e2.timestamp - e1.timestamp).total_seconds() / 3600

                    # Retaliation within 1-48 hours
                    if 1 <= time_diff <= 48:
                        pattern = DetectedPattern(
                            pattern_id=f"RET_{parts[0]}_{parts[1]}_{e1.timestamp.strftime('%Y%m%d%H%M')}",
                            pattern_type=PatternType.RETALIATION,
                            confidence=max(0.5, 1.0 - (time_diff / 48)),
                            start_time=e1.timestamp,
                            end_time=e2.timestamp,
                            countries=[parts[0], parts[1]],
                            description=f"Possible retaliation: {parts[0]} -> {parts[1]} followed by {parts[1]} -> {parts[0]}",
                            severity=min(1.0, (abs(e1.goldstein_scale) + abs(e2.goldstein_scale)) / 20),
                            metrics={
                                "hours_between": time_diff,
                                "initial_goldstein": e1.goldstein_scale,
                                "response_goldstein": e2.goldstein_scale,
                            }
                        )
                        patterns.append(pattern)
                        break

        return patterns


class CoordinationDetector(PatternDetector):
    """Detects coordinated actions between multiple actors."""

    def __init__(self, time_threshold_minutes: int = 60):
        self.time_threshold = timedelta(minutes=time_threshold_minutes)

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.COORDINATION

    def detect(self, events: List[Any],
               time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        patterns = []

        # Group by target (actor2)
        target_events: Dict[str, List[Any]] = defaultdict(list)

        for e in events:
            if e.actor2_country:
                target_events[e.actor2_country].append(e)

        for target, evts in target_events.items():
            if len(evts) < 3:
                continue

            # Sort by time
            evts.sort(key=lambda x: x.timestamp)

            # Look for clusters of actions from different actors
            for i, anchor in enumerate(evts[:-2]):
                cluster = [anchor]
                sources = {anchor.actor1_country}

                for j in range(i + 1, len(evts)):
                    time_diff = evts[j].timestamp - anchor.timestamp

                    if time_diff <= self.time_threshold:
                        if evts[j].actor1_country not in sources:
                            cluster.append(evts[j])
                            sources.add(evts[j].actor1_country)

                # Multiple different actors acting on same target = coordination
                if len(cluster) >= 3 and len(sources) >= 3:
                    # Check if same type of action
                    action_types = {e.event_root for e in cluster}

                    if len(action_types) <= 2:  # Similar actions
                        pattern = DetectedPattern(
                            pattern_id=f"COORD_{target}_{anchor.timestamp.strftime('%Y%m%d%H%M')}",
                            pattern_type=PatternType.COORDINATION,
                            confidence=min(1.0, len(cluster) / 5),
                            start_time=cluster[0].timestamp,
                            end_time=cluster[-1].timestamp,
                            actors=list(sources),
                            countries=[target],
                            description=f"Coordinated action by {', '.join(sources)} against {target}",
                            severity=0.5,
                            metrics={
                                "actor_count": len(sources),
                                "event_count": len(cluster),
                            }
                        )
                        patterns.append(pattern)

        return patterns


class AnomalyDetector(PatternDetector):
    """Detects anomalous patterns using statistical methods."""

    @property
    def pattern_type(self) -> PatternType:
        return PatternType.ANOMALY

    def detect(self, events: List[Any],
               time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        patterns = []

        # Detect unusual actor pairs
        pair_counts: Dict[str, int] = defaultdict(int)
        pair_first_seen: Dict[str, datetime] = {}

        for e in events:
            if e.actor1_country and e.actor2_country:
                key = tuple(sorted([e.actor1_country, e.actor2_country]))
                pair_key = f"{key[0]}:{key[1]}"
                pair_counts[pair_key] += 1
                if pair_key not in pair_first_seen:
                    pair_first_seen[pair_key] = e.timestamp

        # Find pairs that suddenly appeared with high activity
        now = datetime.now(timezone.utc)
        recent_window = now - timedelta(hours=48)

        for pair_key, count in pair_counts.items():
            first = pair_first_seen.get(pair_key)

            if first and first > recent_window and count >= 10:
                countries = pair_key.split(":")
                pattern = DetectedPattern(
                    pattern_id=f"ANOM_{pair_key}_{now.strftime('%Y%m%d%H%M')}",
                    pattern_type=PatternType.ANOMALY,
                    confidence=min(1.0, count / 20),
                    start_time=first,
                    countries=countries,
                    description=f"New high-activity relationship: {countries[0]} - {countries[1]} ({count} events)",
                    severity=min(1.0, count / 30),
                    metrics={
                        "event_count": count,
                        "hours_since_first": (now - first).total_seconds() / 3600,
                    }
                )
                patterns.append(pattern)

        return patterns


# ═══════════════════════════════════════════════════════════════════════════════
# PLUGGABLE ANALYTICS ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════════

class AnalyticsPlugin(ABC):
    """Base class for pluggable analytics modules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass

    @abstractmethod
    def process(self, events: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process events and return results.

        Args:
            events: List of IntelEvent objects
            context: Additional context (registry, previous results, etc.)

        Returns:
            Dictionary of results
        """
        pass


class SentimentAnalyzer(AnalyticsPlugin):
    """Analyzes sentiment trends."""

    @property
    def name(self) -> str:
        return "SentimentAnalyzer"

    @property
    def version(self) -> str:
        return "1.0.0"

    def process(self, events: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        if not events:
            return {"error": "No events to analyze"}

        # Overall sentiment
        tones = [e.avg_tone for e in events if e.avg_tone != 0]
        avg_tone = sum(tones) / len(tones) if tones else 0

        # Sentiment by country
        country_tones: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            if e.actor1_country and e.avg_tone != 0:
                country_tones[e.actor1_country].append(e.avg_tone)

        country_sentiment = {
            country: sum(tones) / len(tones)
            for country, tones in country_tones.items()
            if len(tones) >= 5
        }

        # Trend over time
        hourly_tones: Dict[str, List[float]] = defaultdict(list)
        for e in events:
            hour = e.timestamp.strftime("%Y%m%d%H")
            if e.avg_tone != 0:
                hourly_tones[hour].append(e.avg_tone)

        hourly_avg = {
            hour: sum(tones) / len(tones)
            for hour, tones in sorted(hourly_tones.items())
        }

        return {
            "overall_sentiment": avg_tone,
            "sentiment_label": "negative" if avg_tone < -1 else "positive" if avg_tone > 1 else "neutral",
            "country_sentiment": dict(sorted(country_sentiment.items(), key=lambda x: x[1])),
            "hourly_trend": hourly_avg,
            "most_negative": min(country_sentiment, key=country_sentiment.get) if country_sentiment else None,
            "most_positive": max(country_sentiment, key=country_sentiment.get) if country_sentiment else None,
        }


class GeographicHotspotAnalyzer(AnalyticsPlugin):
    """Identifies geographic hotspots."""

    @property
    def name(self) -> str:
        return "GeographicHotspotAnalyzer"

    @property
    def version(self) -> str:
        return "1.0.0"

    def process(self, events: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # MGRS zone analysis
        mgrs_activity: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "conflict": 0, "avg_goldstein": 0.0, "goldsteins": []}
        )

        for e in events:
            if e.mgrs and len(e.mgrs) >= 5:
                zone = e.mgrs[:5]
                mgrs_activity[zone]["count"] += 1
                mgrs_activity[zone]["goldsteins"].append(e.goldstein_scale)
                if e.is_conflict():
                    mgrs_activity[zone]["conflict"] += 1

        # Calculate averages
        for zone, data in mgrs_activity.items():
            if data["goldsteins"]:
                data["avg_goldstein"] = sum(data["goldsteins"]) / len(data["goldsteins"])
            del data["goldsteins"]

        # Sort by conflict activity
        hotspots = sorted(
            [(zone, data) for zone, data in mgrs_activity.items()],
            key=lambda x: -x[1]["conflict"]
        )[:20]

        return {
            "total_zones": len(mgrs_activity),
            "hotspots": [
                {"zone": z, **d} for z, d in hotspots
            ],
        }


class EventTypeBreakdown(AnalyticsPlugin):
    """Breaks down event types."""

    @property
    def name(self) -> str:
        return "EventTypeBreakdown"

    @property
    def version(self) -> str:
        return "1.0.0"

    def process(self, events: List[Any], context: Dict[str, Any]) -> Dict[str, Any]:
        # Quad class distribution
        quad_counts = defaultdict(int)
        for e in events:
            quad_counts[e.quad_class] += 1

        quad_labels = {
            1: "Verbal Cooperation",
            2: "Material Cooperation",
            3: "Verbal Conflict",
            4: "Material Conflict",
        }

        # Event root codes
        root_counts = defaultdict(int)
        for e in events:
            root_counts[e.event_description] += 1

        return {
            "quad_distribution": {
                quad_labels.get(q, f"Quad {q}"): count
                for q, count in sorted(quad_counts.items())
            },
            "cooperation_ratio": (quad_counts[1] + quad_counts[2]) / len(events) if events else 0,
            "conflict_ratio": (quad_counts[3] + quad_counts[4]) / len(events) if events else 0,
            "top_event_types": dict(sorted(root_counts.items(), key=lambda x: -x[1])[:15]),
        }


class AnalyticsEngine:
    """Pluggable analytics engine."""

    def __init__(self):
        self.plugins: Dict[str, AnalyticsPlugin] = {}
        self.pattern_detectors: Dict[str, PatternDetector] = {}
        self.actor_registry = ActorRegistry()
        self.coalition_detector = CoalitionDetector(self.actor_registry)
        self.results_cache: Dict[str, Any] = {}

        # Register default plugins
        self.register_plugin(SentimentAnalyzer())
        self.register_plugin(GeographicHotspotAnalyzer())
        self.register_plugin(EventTypeBreakdown())

        # Register default pattern detectors
        self.register_pattern_detector(EscalationDetector())
        self.register_pattern_detector(SurgeDetector())
        self.register_pattern_detector(RetaliationDetector())
        self.register_pattern_detector(CoordinationDetector())
        self.register_pattern_detector(AnomalyDetector())

    def register_plugin(self, plugin: AnalyticsPlugin):
        """Register an analytics plugin."""
        self.plugins[plugin.name] = plugin
        logger.info(f"Registered plugin: {plugin.name} v{plugin.version}")

    def unregister_plugin(self, name: str):
        """Unregister a plugin."""
        if name in self.plugins:
            del self.plugins[name]

    def register_pattern_detector(self, detector: PatternDetector):
        """Register a pattern detector."""
        self.pattern_detectors[detector.pattern_type.value] = detector
        logger.info(f"Registered pattern detector: {detector.pattern_type.value}")

    def process_events(self, events: List[Any]) -> None:
        """Process events to build actor registry."""
        for e in events:
            # Register actors
            if e.actor1_code or e.actor1_name:
                actor1 = self.actor_registry.get_or_create_actor(
                    e.actor1_code, e.actor1_name, e.actor1_country, e.actor1_type
                )
                actor1.total_events += 1
                actor1.events_as_actor1 += 1

                if actor1.first_seen is None or e.timestamp < actor1.first_seen:
                    actor1.first_seen = e.timestamp
                if actor1.last_seen is None or e.timestamp > actor1.last_seen:
                    actor1.last_seen = e.timestamp

            if e.actor2_code or e.actor2_name:
                actor2 = self.actor_registry.get_or_create_actor(
                    e.actor2_code, e.actor2_name, e.actor2_country, e.actor2_type
                )
                actor2.total_events += 1
                actor2.events_as_actor2 += 1

            # Register relationship
            if (e.actor1_code or e.actor1_name) and (e.actor2_code or e.actor2_name):
                a1_id = self.actor_registry._generate_actor_id(e.actor1_code, e.actor1_name)
                a2_id = self.actor_registry._generate_actor_id(e.actor2_code, e.actor2_name)

                rel = self.actor_registry.get_or_create_relationship(a1_id, a2_id)
                rel.total_interactions += 1

                if e.quad_class in (1, 2):
                    rel.cooperative_interactions += 1
                elif e.quad_class in (3, 4):
                    rel.conflictual_interactions += 1

                if rel.first_interaction is None:
                    rel.first_interaction = e.timestamp
                rel.last_interaction = e.timestamp

        # Calculate actor metrics
        for actor in self.actor_registry.actors.values():
            if actor.total_events > 0 and actor.first_seen and actor.last_seen:
                days = max(1, (actor.last_seen - actor.first_seen).days)
                actor.activity_level = actor.total_events / days

    def run_plugins(self, events: List[Any]) -> Dict[str, Any]:
        """Run all registered plugins."""
        context = {
            "actor_registry": self.actor_registry,
            "previous_results": self.results_cache,
        }

        results = {}
        for name, plugin in self.plugins.items():
            try:
                results[name] = plugin.process(events, context)
            except Exception as e:
                results[name] = {"error": str(e)}
                logger.error(f"Plugin {name} failed: {e}")

        self.results_cache = results
        return results

    def detect_patterns(self, events: List[Any],
                        time_window: timedelta = timedelta(hours=24)) -> List[DetectedPattern]:
        """Run all pattern detectors."""
        all_patterns = []

        for name, detector in self.pattern_detectors.items():
            try:
                patterns = detector.detect(events, time_window)
                all_patterns.extend(patterns)
            except Exception as e:
                logger.error(f"Pattern detector {name} failed: {e}")

        # Sort by severity and confidence
        all_patterns.sort(key=lambda p: (-(p.severity or 0), -(p.confidence or 0)))

        return all_patterns

    def detect_coalitions(self, min_size: int = 2,
                          min_cooperation: float = 0.6) -> List[ActorCoalition]:
        """Detect actor coalitions."""
        return self.coalition_detector.detect_coalitions(min_size, min_cooperation)

    def get_full_analysis(self, events: List[Any]) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        # Process events
        self.process_events(events)

        # Run plugins
        plugin_results = self.run_plugins(events)

        # Detect patterns
        patterns = self.detect_patterns(events)

        # Detect coalitions
        coalitions = self.detect_coalitions()

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_count": len(events),
            "actor_count": len(self.actor_registry.actors),
            "relationship_count": len(self.actor_registry.relationships),
            "analytics": plugin_results,
            "patterns": [
                {
                    "type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "severity": p.severity,
                    "description": p.description,
                    "actors": p.actors,
                    "countries": p.countries,
                }
                for p in patterns[:20]
            ],
            "coalitions": [
                {
                    "id": c.coalition_id,
                    "name": c.name,
                    "member_count": len(c.members),
                    "cohesion": c.cohesion_score,
                    "dominant_country": c.dominant_country,
                }
                for c in coalitions[:10]
            ],
            "top_actors": [
                {
                    "name": a.name,
                    "country": a.country_code,
                    "type": a.actor_type.value,
                    "events": a.total_events,
                }
                for a in self.actor_registry.get_top_actors(15)
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_analytics_engine: Optional[AnalyticsEngine] = None


def get_analytics_engine() -> AnalyticsEngine:
    """Get global analytics engine instance."""
    global _analytics_engine
    if _analytics_engine is None:
        _analytics_engine = AnalyticsEngine()
    return _analytics_engine


def reset_analytics_engine():
    """Reset the analytics engine."""
    global _analytics_engine
    _analytics_engine = None
