"""
PRISM-WEAVER Intelligence Narrative System

Operational codenames and capabilities:
- STORMGLASS: Real-time event ingestion and correlation
- LOOKING-GLASS: Target profiling and pattern recognition
- THREADNEEDLE: Narrative thread extraction and weaving
- GHOSTWRITER: Automated intelligence report generation
- SHADOWGRAPH: Relationship and network visualization
- CRYSTALBALL: Predictive analytics and forecasting
- NIGHTOWL: Continuous monitoring and alerting
- IRONCLAD: Confidence assessment and source validation

Classification: UNCLASSIFIED // FOR TRAINING PURPOSES ONLY
"""

from __future__ import annotations
import hashlib
import json
import re
import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Set, Optional, Tuple, Any, Callable,
    Protocol, Type, TypeVar, Union, Iterator
)
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION AND HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class ClassLevel(str, Enum):
    """Classification levels."""
    UNCLASSIFIED = "UNCLASSIFIED"
    RESTRICTED = "RESTRICTED"
    CONFIDENTIAL = "CONFIDENTIAL"
    SECRET = "SECRET"
    TOP_SECRET = "TOP SECRET"


class Compartment(str, Enum):
    """Compartmentalized access controls."""
    SIGINT = "SI"           # Signals Intelligence
    HUMINT = "HU"           # Human Intelligence
    OSINT = "OS"            # Open Source Intelligence
    GEOINT = "GI"           # Geospatial Intelligence
    MASINT = "MA"           # Measurement and Signature Intelligence
    FININT = "FI"           # Financial Intelligence
    CYBINT = "CY"           # Cyber Intelligence


class Caveat(str, Enum):
    """Dissemination caveats."""
    NOFORN = "NOFORN"           # No Foreign Nationals
    FVEY = "FVEY"               # Five Eyes Only
    REL_TO = "REL TO"           # Releasable To
    ORCON = "ORCON"             # Originator Controlled
    PROPIN = "PROPIN"           # Proprietary Information
    NOCONTRACT = "NOCONTRACT"   # Not Releasable to Contractors


@dataclass
class ClassificationMarking:
    """Full classification marking."""
    level: ClassLevel = ClassLevel.UNCLASSIFIED
    compartments: List[Compartment] = field(default_factory=list)
    caveats: List[Caveat] = field(default_factory=list)
    codeword: str = ""

    def __str__(self) -> str:
        parts = [self.level.value]
        if self.codeword:
            parts.append(f"//{self.codeword}")
        if self.compartments:
            parts.append("//" + "/".join(c.value for c in self.compartments))
        if self.caveats:
            parts.append("//" + "/".join(c.value for c in self.caveats))
        return "".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# STORMGLASS: Event Ingestion and Correlation
# ═══════════════════════════════════════════════════════════════════════════════

class EventSignificance(str, Enum):
    """Event significance levels."""
    ROUTINE = "routine"
    NOTEWORTHY = "noteworthy"
    SIGNIFICANT = "significant"
    CRITICAL = "critical"
    FLASH = "flash"


@dataclass
class IntelSelector:
    """A selector for targeting/tracking."""
    selector_id: str
    selector_type: str  # COUNTRY, ACTOR, ORGANIZATION, TOPIC, LOCATION, KEYWORD
    value: str
    priority: int = 3   # 1=highest, 5=lowest
    active: bool = True
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hits: int = 0
    last_hit: Optional[datetime] = None

    def matches(self, event: Any) -> bool:
        """Check if event matches this selector."""
        if self.selector_type == "COUNTRY":
            return (event.actor1_country == self.value or
                    event.actor2_country == self.value)
        elif self.selector_type == "ACTOR":
            return (self.value.lower() in event.actor1_name.lower() or
                    self.value.lower() in event.actor2_name.lower())
        elif self.selector_type == "KEYWORD":
            return self.value.lower() in event.event_description.lower()
        elif self.selector_type == "EVENT_CODE":
            return event.event_code.startswith(self.value)
        return False


@dataclass
class CorrelatedEvent:
    """An event with correlation metadata."""
    event: Any
    selectors_matched: List[str] = field(default_factory=list)
    significance: EventSignificance = EventSignificance.ROUTINE
    correlation_ids: Set[str] = field(default_factory=set)
    narrative_threads: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)

    @property
    def correlation_score(self) -> float:
        """Calculate correlation importance."""
        base = len(self.selectors_matched) * 2
        base += len(self.correlation_ids)
        base += {"routine": 0, "noteworthy": 1, "significant": 3,
                 "critical": 5, "flash": 10}[self.significance.value]
        return min(10.0, base)


class STORMGLASS:
    """
    Real-time event ingestion and correlation engine.

    Ingests raw events, matches against selectors, and correlates
    related events into trackable threads.
    """

    def __init__(self):
        self.selectors: Dict[str, IntelSelector] = {}
        self.correlated_events: List[CorrelatedEvent] = []
        self.correlation_index: Dict[str, List[int]] = defaultdict(list)
        self.event_count = 0
        self.last_ingest: Optional[datetime] = None

    def add_selector(self, selector: IntelSelector) -> str:
        """Add a new selector for matching."""
        self.selectors[selector.selector_id] = selector
        return selector.selector_id

    def create_selector(self, selector_type: str, value: str,
                        priority: int = 3) -> IntelSelector:
        """Create and register a new selector."""
        sel_id = f"SEL-{hashlib.md5(f'{selector_type}:{value}'.encode()).hexdigest()[:8].upper()}"
        selector = IntelSelector(
            selector_id=sel_id,
            selector_type=selector_type,
            value=value,
            priority=priority,
        )
        self.add_selector(selector)
        return selector

    def ingest(self, events: List[Any]) -> int:
        """Ingest events and correlate."""
        new_correlations = 0

        for event in events:
            ce = CorrelatedEvent(event=event)

            # Match against selectors
            for sel_id, selector in self.selectors.items():
                if selector.active and selector.matches(event):
                    ce.selectors_matched.append(sel_id)
                    selector.hits += 1
                    selector.last_hit = event.timestamp

            # Determine significance
            if len(ce.selectors_matched) >= 3:
                ce.significance = EventSignificance.SIGNIFICANT
            elif len(ce.selectors_matched) >= 2:
                ce.significance = EventSignificance.NOTEWORTHY
            elif event.goldstein_scale <= -7:
                ce.significance = EventSignificance.CRITICAL
            elif event.goldstein_scale <= -5:
                ce.significance = EventSignificance.SIGNIFICANT

            # Build correlation IDs
            if event.actor1_country and event.actor2_country:
                dyad = tuple(sorted([event.actor1_country, event.actor2_country]))
                ce.correlation_ids.add(f"DYAD:{dyad[0]}-{dyad[1]}")

            if event.actor1_country:
                ce.correlation_ids.add(f"COUNTRY:{event.actor1_country}")
            if event.actor2_country:
                ce.correlation_ids.add(f"COUNTRY:{event.actor2_country}")

            ce.correlation_ids.add(f"EVTYPE:{event.event_root}")

            # Index for retrieval
            idx = len(self.correlated_events)
            self.correlated_events.append(ce)

            for cid in ce.correlation_ids:
                self.correlation_index[cid].append(idx)

            if ce.selectors_matched:
                new_correlations += 1

        self.event_count += len(events)
        self.last_ingest = datetime.now(timezone.utc)
        return new_correlations

    def query(self, correlation_id: str) -> List[CorrelatedEvent]:
        """Query events by correlation ID."""
        indices = self.correlation_index.get(correlation_id, [])
        return [self.correlated_events[i] for i in indices]

    def get_significant_events(self,
                               min_significance: EventSignificance = EventSignificance.NOTEWORTHY,
                               limit: int = 100) -> List[CorrelatedEvent]:
        """Get events above significance threshold."""
        levels = {
            EventSignificance.ROUTINE: 0,
            EventSignificance.NOTEWORTHY: 1,
            EventSignificance.SIGNIFICANT: 2,
            EventSignificance.CRITICAL: 3,
            EventSignificance.FLASH: 4,
        }
        min_level = levels[min_significance]

        significant = [
            ce for ce in self.correlated_events
            if levels[ce.significance] >= min_level
        ]

        return sorted(significant, key=lambda x: -x.correlation_score)[:limit]


# ═══════════════════════════════════════════════════════════════════════════════
# LOOKING-GLASS: Target Profiling
# ═══════════════════════════════════════════════════════════════════════════════

class TargetType(str, Enum):
    """Types of intelligence targets."""
    STATE_ACTOR = "state_actor"
    NON_STATE_ACTOR = "non_state_actor"
    ORGANIZATION = "organization"
    INDIVIDUAL = "individual"
    LOCATION = "location"
    NETWORK = "network"
    TOPIC = "topic"


class ThreatVector(str, Enum):
    """Threat vectors."""
    MILITARY = "military"
    POLITICAL = "political"
    ECONOMIC = "economic"
    CYBER = "cyber"
    TERRORISM = "terrorism"
    CRIMINAL = "criminal"
    INFLUENCE = "influence"
    WMD = "wmd"


@dataclass
class TargetPackage:
    """
    Comprehensive target intelligence package.

    Contains all collected intelligence on a specific target,
    suitable for briefing or operational planning.
    """
    target_id: str
    codename: str
    target_type: TargetType

    # Basic info
    name: str
    aliases: Set[str] = field(default_factory=set)
    country: str = ""
    description: str = ""

    # Classification
    classification: ClassificationMarking = field(
        default_factory=lambda: ClassificationMarking(ClassLevel.UNCLASSIFIED)
    )

    # Threat assessment
    threat_vectors: List[ThreatVector] = field(default_factory=list)
    threat_level: int = 0  # 1-10
    priority: int = 3      # 1-5, 1=highest

    # Activity metrics
    total_events: int = 0
    first_observed: Optional[datetime] = None
    last_observed: Optional[datetime] = None
    activity_trend: str = "stable"  # increasing, stable, decreasing

    # Behavioral profile
    aggression_index: float = 0.0      # 0-1
    cooperation_index: float = 0.0     # 0-1
    volatility_index: float = 0.0      # 0-1
    influence_index: float = 0.0       # 0-1

    # Relationships
    allies: List[Tuple[str, float, str]] = field(default_factory=list)       # (target_id, strength, basis)
    adversaries: List[Tuple[str, float, str]] = field(default_factory=list)
    associates: List[Tuple[str, float, str]] = field(default_factory=list)

    # Event breakdown
    event_distribution: Dict[str, int] = field(default_factory=dict)
    quad_distribution: Dict[int, int] = field(default_factory=dict)

    # Geographic footprint
    locations_active: List[Tuple[str, int]] = field(default_factory=list)  # (location, event_count)
    mgrs_zones: List[str] = field(default_factory=list)

    # Key events
    significant_events: List[Any] = field(default_factory=list)

    # Assessments
    analyst_notes: List[Tuple[datetime, str, str]] = field(default_factory=list)  # (time, analyst, note)

    # Selectors
    active_selectors: List[str] = field(default_factory=list)

    def get_threat_summary(self) -> str:
        """Generate threat summary."""
        vectors = ", ".join(v.value for v in self.threat_vectors) if self.threat_vectors else "None identified"
        return f"THREAT LEVEL: {self.threat_level}/10 | VECTORS: {vectors}"

    def get_activity_summary(self) -> str:
        """Generate activity summary."""
        if self.first_observed and self.last_observed:
            duration = (self.last_observed - self.first_observed).days
            rate = self.total_events / max(1, duration) if duration > 0 else self.total_events
            return f"{self.total_events} events over {duration} days ({rate:.1f}/day, {self.activity_trend})"
        return f"{self.total_events} events"


class LOOKING_GLASS:
    """
    Target profiling and pattern recognition system.

    Builds comprehensive target packages from raw event data,
    identifying patterns, relationships, and behavioral signatures.
    """

    # Codename generators
    ADJECTIVES = [
        "SILENT", "SHADOW", "IRON", "CRYSTAL", "MIDNIGHT", "ARCTIC", "DESERT",
        "OCEAN", "THUNDER", "LIGHTNING", "PHANTOM", "GHOST", "STEEL", "GOLDEN",
        "CRIMSON", "AZURE", "EMERALD", "OBSIDIAN", "SILVER", "COPPER", "BRASS"
    ]
    NOUNS = [
        "FALCON", "EAGLE", "WOLF", "BEAR", "LION", "TIGER", "SHARK", "DRAGON",
        "PHOENIX", "VIPER", "COBRA", "HAWK", "RAVEN", "CONDOR", "SERPENT",
        "PANTHER", "LEOPARD", "JAGUAR", "RAPTOR", "HYDRA", "CHIMERA", "SPHINX"
    ]

    def __init__(self):
        self.targets: Dict[str, TargetPackage] = {}
        self.codename_map: Dict[str, str] = {}  # codename -> target_id
        self._codename_counter = 0

    def _generate_codename(self, seed: str = "") -> str:
        """Generate unique codename."""
        import random
        if seed:
            random.seed(hash(seed) % 2**32)

        while True:
            adj = random.choice(self.ADJECTIVES)
            noun = random.choice(self.NOUNS)
            codename = f"{adj}-{noun}"
            if codename not in self.codename_map:
                return codename
            self._codename_counter += 1
            codename = f"{adj}-{noun}-{self._codename_counter}"
            if codename not in self.codename_map:
                return codename

    def create_target(self, name: str, target_type: TargetType,
                      country: str = "") -> TargetPackage:
        """Create a new target package."""
        target_id = f"TGT-{hashlib.md5(name.encode()).hexdigest()[:8].upper()}"
        codename = self._generate_codename(name)

        target = TargetPackage(
            target_id=target_id,
            codename=codename,
            target_type=target_type,
            name=name,
            country=country,
        )

        self.targets[target_id] = target
        self.codename_map[codename] = target_id
        return target

    def get_target(self, identifier: str) -> Optional[TargetPackage]:
        """Get target by ID or codename."""
        if identifier in self.targets:
            return self.targets[identifier]
        if identifier in self.codename_map:
            return self.targets[self.codename_map[identifier]]
        return None

    def build_target_from_events(self, name: str, events: List[Any],
                                  target_type: TargetType = TargetType.STATE_ACTOR) -> TargetPackage:
        """Build target package from event stream."""
        # Filter relevant events
        relevant = [e for e in events if
                    name.lower() in e.actor1_name.lower() or
                    name.lower() in e.actor2_name.lower() or
                    e.actor1_country == name or e.actor2_country == name]

        if not relevant:
            return self.create_target(name, target_type)

        target = self.create_target(name, target_type)
        target.total_events = len(relevant)

        # Time range
        timestamps = [e.timestamp for e in relevant]
        target.first_observed = min(timestamps)
        target.last_observed = max(timestamps)

        # Behavioral metrics
        goldsteins = [e.goldstein_scale for e in relevant]
        quad_classes = [e.quad_class for e in relevant]

        conflict_events = sum(1 for q in quad_classes if q in (3, 4))
        coop_events = sum(1 for q in quad_classes if q in (1, 2))

        target.aggression_index = conflict_events / len(relevant) if relevant else 0
        target.cooperation_index = coop_events / len(relevant) if relevant else 0

        # Volatility (standard deviation of Goldstein)
        if goldsteins:
            mean_g = sum(goldsteins) / len(goldsteins)
            variance = sum((g - mean_g) ** 2 for g in goldsteins) / len(goldsteins)
            target.volatility_index = min(1.0, math.sqrt(variance) / 10)

        # Event distribution
        for e in relevant:
            target.event_distribution[e.event_root] = target.event_distribution.get(e.event_root, 0) + 1
            target.quad_distribution[e.quad_class] = target.quad_distribution.get(e.quad_class, 0) + 1

        # Relationships
        partner_events: Dict[str, List[Any]] = defaultdict(list)
        adversary_events: Dict[str, List[Any]] = defaultdict(list)

        for e in relevant:
            other = None
            if e.actor1_country == name or name.lower() in e.actor1_name.lower():
                other = e.actor2_country or e.actor2_name
            else:
                other = e.actor1_country or e.actor1_name

            if other:
                if e.quad_class in (1, 2):
                    partner_events[other].append(e)
                elif e.quad_class in (3, 4):
                    adversary_events[other].append(e)

        # Build relationship lists
        for other, evts in partner_events.items():
            strength = len(evts) / len(relevant)
            avg_g = sum(e.goldstein_scale for e in evts) / len(evts)
            target.allies.append((other, strength, f"{len(evts)} cooperative events, avg Goldstein {avg_g:+.1f}"))
        target.allies.sort(key=lambda x: -x[1])

        for other, evts in adversary_events.items():
            strength = len(evts) / len(relevant)
            avg_g = sum(e.goldstein_scale for e in evts) / len(evts)
            target.adversaries.append((other, strength, f"{len(evts)} conflict events, avg Goldstein {avg_g:+.1f}"))
        target.adversaries.sort(key=lambda x: -x[1])

        # Threat assessment
        if target.aggression_index > 0.5:
            target.threat_level = min(10, int(target.aggression_index * 10) + 3)
            if any(e.event_root in ("18", "19", "20") for e in relevant):
                target.threat_vectors.append(ThreatVector.MILITARY)
            if any(e.event_root in ("17",) for e in relevant):
                target.threat_vectors.append(ThreatVector.POLITICAL)

        # Significant events
        target.significant_events = sorted(
            [e for e in relevant if e.goldstein_scale <= -5 or e.num_mentions >= 20],
            key=lambda x: x.goldstein_scale
        )[:10]

        # Activity trend
        if len(relevant) >= 10:
            mid = len(relevant) // 2
            sorted_by_time = sorted(relevant, key=lambda x: x.timestamp)
            early_count = mid
            late_count = len(relevant) - mid

            early_duration = (sorted_by_time[mid-1].timestamp - sorted_by_time[0].timestamp).days or 1
            late_duration = (sorted_by_time[-1].timestamp - sorted_by_time[mid].timestamp).days or 1

            early_rate = early_count / early_duration
            late_rate = late_count / late_duration

            if late_rate > early_rate * 1.5:
                target.activity_trend = "increasing"
            elif late_rate < early_rate * 0.67:
                target.activity_trend = "decreasing"
            else:
                target.activity_trend = "stable"

        return target

    def generate_target_brief(self, target: TargetPackage) -> str:
        """Generate formatted target brief."""
        lines = [
            f"{'═' * 80}",
            f"TARGET INTELLIGENCE PACKAGE",
            f"{'═' * 80}",
            f"",
            f"CODENAME: {target.codename}",
            f"TARGET ID: {target.target_id}",
            f"CLASSIFICATION: {target.classification}",
            f"",
            f"{'─' * 80}",
            f"BASIC INFORMATION",
            f"{'─' * 80}",
            f"Name: {target.name}",
            f"Type: {target.target_type.value}",
            f"Country: {target.country or 'N/A'}",
            f"Aliases: {', '.join(target.aliases) if target.aliases else 'None known'}",
            f"",
            f"{'─' * 80}",
            f"THREAT ASSESSMENT",
            f"{'─' * 80}",
            f"Threat Level: {target.threat_level}/10",
            f"Priority: {target.priority}/5",
            f"Threat Vectors: {', '.join(v.value for v in target.threat_vectors) if target.threat_vectors else 'None identified'}",
            f"",
            f"{'─' * 80}",
            f"ACTIVITY PROFILE",
            f"{'─' * 80}",
            f"Total Events: {target.total_events:,}",
            f"First Observed: {target.first_observed.strftime('%Y-%m-%d %H:%M UTC') if target.first_observed else 'N/A'}",
            f"Last Observed: {target.last_observed.strftime('%Y-%m-%d %H:%M UTC') if target.last_observed else 'N/A'}",
            f"Activity Trend: {target.activity_trend.upper()}",
            f"",
            f"{'─' * 80}",
            f"BEHAVIORAL INDICES",
            f"{'─' * 80}",
            f"Aggression Index:  {'█' * int(target.aggression_index * 20)}{'░' * (20 - int(target.aggression_index * 20))} {target.aggression_index:.2f}",
            f"Cooperation Index: {'█' * int(target.cooperation_index * 20)}{'░' * (20 - int(target.cooperation_index * 20))} {target.cooperation_index:.2f}",
            f"Volatility Index:  {'█' * int(target.volatility_index * 20)}{'░' * (20 - int(target.volatility_index * 20))} {target.volatility_index:.2f}",
            f"",
        ]

        if target.allies:
            lines.extend([
                f"{'─' * 80}",
                f"KEY RELATIONSHIPS - COOPERATIVE",
                f"{'─' * 80}",
            ])
            for ally, strength, basis in target.allies[:5]:
                lines.append(f"  • {ally}: {basis}")

        if target.adversaries:
            lines.extend([
                f"",
                f"{'─' * 80}",
                f"KEY RELATIONSHIPS - ADVERSARIAL",
                f"{'─' * 80}",
            ])
            for adv, strength, basis in target.adversaries[:5]:
                lines.append(f"  • {adv}: {basis}")

        if target.significant_events:
            lines.extend([
                f"",
                f"{'─' * 80}",
                f"SIGNIFICANT EVENTS",
                f"{'─' * 80}",
            ])
            for event in target.significant_events[:5]:
                lines.append(f"  • [{event.timestamp.strftime('%Y-%m-%d')}] {event.actor1_name} → {event.actor2_name}")
                lines.append(f"    {event.event_description} (Goldstein: {event.goldstein_scale:+.1f})")

        lines.extend([
            f"",
            f"{'═' * 80}",
            f"END TARGET PACKAGE // {target.codename}",
            f"{'═' * 80}",
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# THREADNEEDLE: Narrative Thread Extraction
# ═══════════════════════════════════════════════════════════════════════════════

class NarrativeType(str, Enum):
    """Types of intelligence narratives."""
    CONFLICT_ESCALATION = "conflict_escalation"
    DIPLOMATIC_INITIATIVE = "diplomatic_initiative"
    MILITARY_OPERATION = "military_operation"
    HUMANITARIAN_CRISIS = "humanitarian_crisis"
    POLITICAL_TRANSITION = "political_transition"
    ECONOMIC_PRESSURE = "economic_pressure"
    ALLIANCE_SHIFT = "alliance_shift"
    COVERT_ACTION = "covert_action"
    INFLUENCE_CAMPAIGN = "influence_campaign"


@dataclass
class NarrativeThread:
    """
    A coherent narrative thread extracted from events.

    Represents a story that can be told from the intelligence,
    connecting events into a meaningful sequence.
    """
    thread_id: str
    title: str
    narrative_type: NarrativeType

    # Time bounds
    start_time: datetime
    end_time: datetime

    # Actors
    primary_actors: List[str] = field(default_factory=list)
    secondary_actors: List[str] = field(default_factory=list)

    # Geographic scope
    countries: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)

    # Events
    events: List[Any] = field(default_factory=list)
    key_events: List[Any] = field(default_factory=list)

    # Narrative elements
    summary: str = ""
    background: str = ""
    developments: List[str] = field(default_factory=list)
    outlook: str = ""

    # Assessment
    confidence: float = 0.0
    significance: EventSignificance = EventSignificance.NOTEWORTHY

    # Metadata
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    analyst: str = "AUTOMATED"

    def get_timeline(self) -> List[Tuple[datetime, str]]:
        """Get event timeline."""
        return sorted([
            (e.timestamp, f"{e.actor1_name} → {e.actor2_name}: {e.event_description}")
            for e in self.events
        ], key=lambda x: x[0])


class THREADNEEDLE:
    """
    Narrative thread extraction and weaving system.

    Identifies coherent narratives from event streams and
    constructs intelligence stories.
    """

    NARRATIVE_INDICATORS = {
        NarrativeType.CONFLICT_ESCALATION: {
            "event_roots": ["13", "14", "15", "17", "18", "19", "20"],
            "goldstein_threshold": -3,
            "min_events": 5,
        },
        NarrativeType.DIPLOMATIC_INITIATIVE: {
            "event_roots": ["03", "04", "05", "06"],
            "goldstein_threshold": 3,
            "min_events": 3,
        },
        NarrativeType.MILITARY_OPERATION: {
            "event_roots": ["15", "18", "19", "20"],
            "goldstein_threshold": -5,
            "min_events": 3,
        },
        NarrativeType.HUMANITARIAN_CRISIS: {
            "event_roots": ["07", "20"],
            "keywords": ["refugee", "humanitarian", "crisis", "disaster"],
            "min_events": 3,
        },
    }

    def __init__(self):
        self.threads: Dict[str, NarrativeThread] = {}
        self._thread_counter = 0

    def _generate_thread_id(self) -> str:
        self._thread_counter += 1
        return f"THREAD-{self._thread_counter:04d}"

    def extract_narratives(self, events: List[Any],
                           time_window: timedelta = timedelta(days=7)) -> List[NarrativeThread]:
        """Extract narrative threads from events."""
        narratives = []

        # Group events by actor pair and time
        dyad_events: Dict[str, List[Any]] = defaultdict(list)

        for e in events:
            if e.actor1_country and e.actor2_country:
                dyad = tuple(sorted([e.actor1_country, e.actor2_country]))
                dyad_events[f"{dyad[0]}-{dyad[1]}"].append(e)

        # Look for narrative patterns in each dyad
        for dyad_key, dyad_evts in dyad_events.items():
            if len(dyad_evts) < 3:
                continue

            dyad_evts.sort(key=lambda x: x.timestamp)

            # Check for conflict escalation
            conflict_evts = [e for e in dyad_evts if e.quad_class in (3, 4)]
            if len(conflict_evts) >= 5:
                goldsteins = [e.goldstein_scale for e in conflict_evts]

                # Check for negative trend
                if len(goldsteins) >= 3:
                    trend = (goldsteins[-1] - goldsteins[0]) / len(goldsteins)

                    if trend < -0.5 or min(goldsteins) <= -5:
                        narrative = self._build_conflict_narrative(dyad_key, conflict_evts)
                        if narrative:
                            narratives.append(narrative)

            # Check for diplomatic initiative
            coop_evts = [e for e in dyad_evts if e.quad_class in (1, 2)]
            if len(coop_evts) >= 3:
                goldsteins = [e.goldstein_scale for e in coop_evts]

                if max(goldsteins) >= 5:
                    narrative = self._build_diplomatic_narrative(dyad_key, coop_evts)
                    if narrative:
                        narratives.append(narrative)

        # Sort by significance
        narratives.sort(key=lambda n: -{"routine": 0, "noteworthy": 1, "significant": 2,
                                        "critical": 3, "flash": 4}[n.significance.value])

        for n in narratives:
            self.threads[n.thread_id] = n

        return narratives

    def _build_conflict_narrative(self, dyad_key: str, events: List[Any]) -> Optional[NarrativeThread]:
        """Build a conflict escalation narrative."""
        if not events:
            return None

        countries = dyad_key.split("-")

        thread = NarrativeThread(
            thread_id=self._generate_thread_id(),
            title=f"ESCALATION: {countries[0]}-{countries[1]} Tensions",
            narrative_type=NarrativeType.CONFLICT_ESCALATION,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            primary_actors=countries,
            countries=countries,
            events=events,
        )

        # Identify key events
        thread.key_events = sorted(events, key=lambda x: x.goldstein_scale)[:5]

        # Build summary
        duration = (thread.end_time - thread.start_time).days
        avg_goldstein = sum(e.goldstein_scale for e in events) / len(events)

        thread.summary = (
            f"Escalating tensions observed between {countries[0]} and {countries[1]} "
            f"over {duration} days. {len(events)} conflict-related events recorded "
            f"with average Goldstein score of {avg_goldstein:.1f}."
        )

        # Build developments
        thread.developments = []
        for event in thread.key_events[:3]:
            thread.developments.append(
                f"[{event.timestamp.strftime('%Y-%m-%d')}] {event.actor1_name} {event.event_description.lower()} "
                f"targeting {event.actor2_name}. (Goldstein: {event.goldstein_scale:+.1f})"
            )

        # Assessment
        if min(e.goldstein_scale for e in events) <= -7:
            thread.significance = EventSignificance.CRITICAL
        elif min(e.goldstein_scale for e in events) <= -5:
            thread.significance = EventSignificance.SIGNIFICANT
        else:
            thread.significance = EventSignificance.NOTEWORTHY

        thread.confidence = min(1.0, len(events) / 10)

        return thread

    def _build_diplomatic_narrative(self, dyad_key: str, events: List[Any]) -> Optional[NarrativeThread]:
        """Build a diplomatic initiative narrative."""
        if not events:
            return None

        countries = dyad_key.split("-")

        thread = NarrativeThread(
            thread_id=self._generate_thread_id(),
            title=f"DIPLOMATIC: {countries[0]}-{countries[1]} Engagement",
            narrative_type=NarrativeType.DIPLOMATIC_INITIATIVE,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            primary_actors=countries,
            countries=countries,
            events=events,
        )

        thread.key_events = sorted(events, key=lambda x: -x.goldstein_scale)[:5]

        duration = (thread.end_time - thread.start_time).days
        avg_goldstein = sum(e.goldstein_scale for e in events) / len(events)

        thread.summary = (
            f"Diplomatic engagement observed between {countries[0]} and {countries[1]} "
            f"over {duration} days. {len(events)} cooperative events recorded "
            f"with average Goldstein score of {avg_goldstein:+.1f}."
        )

        thread.significance = EventSignificance.NOTEWORTHY
        if max(e.goldstein_scale for e in events) >= 7:
            thread.significance = EventSignificance.SIGNIFICANT

        thread.confidence = min(1.0, len(events) / 8)

        return thread

    def weave_narrative(self, thread: NarrativeThread) -> str:
        """Weave a narrative thread into prose."""
        lines = [
            f"{'═' * 80}",
            f"INTELLIGENCE NARRATIVE",
            f"{'═' * 80}",
            f"",
            f"THREAD ID: {thread.thread_id}",
            f"TYPE: {thread.narrative_type.value.upper().replace('_', ' ')}",
            f"TITLE: {thread.title}",
            f"SIGNIFICANCE: {thread.significance.value.upper()}",
            f"CONFIDENCE: {thread.confidence:.0%}",
            f"",
            f"TIME FRAME: {thread.start_time.strftime('%Y-%m-%d')} to {thread.end_time.strftime('%Y-%m-%d')}",
            f"PRIMARY ACTORS: {', '.join(thread.primary_actors)}",
            f"",
            f"{'─' * 80}",
            f"EXECUTIVE SUMMARY",
            f"{'─' * 80}",
            f"",
            thread.summary,
            f"",
        ]

        if thread.developments:
            lines.extend([
                f"{'─' * 80}",
                f"KEY DEVELOPMENTS",
                f"{'─' * 80}",
                f"",
            ])
            for dev in thread.developments:
                lines.append(f"• {dev}")
            lines.append("")

        if thread.key_events:
            lines.extend([
                f"{'─' * 80}",
                f"TIMELINE OF KEY EVENTS",
                f"{'─' * 80}",
                f"",
            ])
            for event in thread.key_events:
                lines.append(f"  {event.timestamp.strftime('%Y-%m-%d %H:%M')} │ {event.actor1_name} → {event.actor2_name}")
                lines.append(f"  {'':19} │ {event.event_description}")
                lines.append(f"  {'':19} │ Goldstein: {event.goldstein_scale:+.1f}, Mentions: {event.num_mentions}")
                lines.append("")

        if thread.outlook:
            lines.extend([
                f"{'─' * 80}",
                f"OUTLOOK",
                f"{'─' * 80}",
                f"",
                thread.outlook,
                f"",
            ])

        lines.extend([
            f"{'═' * 80}",
            f"END NARRATIVE // {thread.thread_id}",
            f"{'═' * 80}",
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# GHOSTWRITER: Automated Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

class ReportType(str, Enum):
    """Types of intelligence reports."""
    DAILY_BRIEF = "daily_brief"
    SITUATION_REPORT = "sitrep"
    THREAT_ASSESSMENT = "threat_assessment"
    TARGET_PACKAGE = "target_package"
    NARRATIVE_SUMMARY = "narrative_summary"
    FLASH_REPORT = "flash_report"
    PERIODIC_SUMMARY = "periodic_summary"


@dataclass
class IntelReport:
    """An intelligence report."""
    report_id: str
    report_type: ReportType
    title: str

    # Classification
    classification: ClassificationMarking = field(
        default_factory=lambda: ClassificationMarking(ClassLevel.UNCLASSIFIED)
    )

    # Metadata
    generated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    analyst: str = "GHOSTWRITER//AUTOMATED"

    # Content sections
    executive_summary: str = ""
    key_judgments: List[str] = field(default_factory=list)
    body_sections: List[Tuple[str, str]] = field(default_factory=list)  # (heading, content)

    # Sources
    source_count: int = 0
    event_count: int = 0

    # Assessment
    confidence_level: str = "MODERATE"  # LOW, MODERATE, HIGH

    def render(self) -> str:
        """Render report to text."""
        lines = [
            f"┏{'━' * 78}┓",
            f"┃{self.classification!s:^78}┃",
            f"┣{'━' * 78}┫",
            f"┃{self.title:^78}┃",
            f"┃{self.report_type.value.upper():^78}┃",
            f"┣{'━' * 78}┫",
            f"┃ Report ID: {self.report_id:64} ┃",
            f"┃ Generated: {self.generated.strftime('%Y-%m-%d %H:%M UTC'):64} ┃",
        ]

        if self.period_start and self.period_end:
            period = f"{self.period_start.strftime('%Y-%m-%d')} to {self.period_end.strftime('%Y-%m-%d')}"
            lines.append(f"┃ Period: {period:67} ┃")

        lines.append(f"┃ Analyst: {self.analyst:66} ┃")
        lines.append(f"┗{'━' * 78}┛")
        lines.append("")

        if self.executive_summary:
            lines.extend([
                "EXECUTIVE SUMMARY",
                "─" * 40,
                "",
                self.executive_summary,
                "",
            ])

        if self.key_judgments:
            lines.extend([
                "KEY JUDGMENTS",
                "─" * 40,
                "",
            ])
            for i, judgment in enumerate(self.key_judgments, 1):
                lines.append(f"  {i}. {judgment}")
            lines.append("")

        for heading, content in self.body_sections:
            lines.extend([
                heading.upper(),
                "─" * 40,
                "",
                content,
                "",
            ])

        lines.extend([
            "─" * 80,
            f"Sources: {self.source_count} | Events: {self.event_count} | Confidence: {self.confidence_level}",
            f"Classification: {self.classification}",
            "─" * 80,
        ])

        return "\n".join(lines)


class GHOSTWRITER:
    """
    Automated intelligence report generation system.

    Generates various types of intelligence reports from
    raw data, narratives, and target packages.
    """

    def __init__(self):
        self.reports: Dict[str, IntelReport] = {}
        self._report_counter = 0

    def _generate_report_id(self, report_type: ReportType) -> str:
        self._report_counter += 1
        prefix = {
            ReportType.DAILY_BRIEF: "DB",
            ReportType.SITUATION_REPORT: "SR",
            ReportType.THREAT_ASSESSMENT: "TA",
            ReportType.TARGET_PACKAGE: "TP",
            ReportType.NARRATIVE_SUMMARY: "NS",
            ReportType.FLASH_REPORT: "FL",
            ReportType.PERIODIC_SUMMARY: "PS",
        }.get(report_type, "IR")

        return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{self._report_counter:04d}"

    def generate_daily_brief(self, events: List[Any],
                             narratives: List[NarrativeThread] = None,
                             targets: List[TargetPackage] = None) -> IntelReport:
        """Generate daily intelligence brief."""
        report = IntelReport(
            report_id=self._generate_report_id(ReportType.DAILY_BRIEF),
            report_type=ReportType.DAILY_BRIEF,
            title="DAILY INTELLIGENCE BRIEF",
        )

        now = datetime.now(timezone.utc)
        report.period_start = now - timedelta(days=1)
        report.period_end = now
        report.event_count = len(events)

        # Helper to get attr from event or wrapped event
        def _get(e, attr, default=None):
            if hasattr(e, attr):
                return getattr(e, attr, default)
            if hasattr(e, 'event') and hasattr(e.event, attr):
                return getattr(e.event, attr, default)
            return default

        # Executive summary
        total_events = len(events)
        conflict_events = sum(1 for e in events if _get(e, 'quad_class', 0) in (3, 4))
        coop_events = sum(1 for e in events if _get(e, 'quad_class', 0) in (1, 2))

        tones = [_get(e, 'avg_tone', 0) for e in events if _get(e, 'avg_tone', 0) != 0]
        avg_tone = sum(tones) / len(tones) if tones else 0

        report.executive_summary = (
            f"In the past 24 hours, {total_events:,} events were recorded globally. "
            f"Cooperative activities ({coop_events:,} events, {100*coop_events/total_events:.0f}%) "
            f"outpaced conflictual activities ({conflict_events:,} events, {100*conflict_events/total_events:.0f}%). "
            f"Overall global sentiment index stands at {avg_tone:+.2f}."
        )

        # Key judgments
        report.key_judgments = []

        # Find most active countries
        country_events: Dict[str, int] = defaultdict(int)
        for e in events:
            c = _get(e, 'actor1_country')
            if c:
                country_events[c] += 1

        top_countries = sorted(country_events.items(), key=lambda x: -x[1])[:5]
        report.key_judgments.append(
            f"Highest activity: {', '.join(f'{c} ({n})' for c, n in top_countries)}"
        )

        # Find emerging conflicts
        conflict_pairs: Dict[str, int] = defaultdict(int)
        for e in events:
            qc = _get(e, 'quad_class', 0)
            a1 = _get(e, 'actor1_country')
            a2 = _get(e, 'actor2_country')
            if qc in (3, 4) and a1 and a2:
                pair = tuple(sorted([a1, a2]))
                conflict_pairs[f"{pair[0]}-{pair[1]}"] += 1

        if conflict_pairs:
            top_conflict = max(conflict_pairs.items(), key=lambda x: x[1])
            report.key_judgments.append(
                f"Primary conflict axis: {top_conflict[0]} ({top_conflict[1]} events)"
            )

        # Add narratives if available
        if narratives:
            critical = [n for n in narratives if n.significance in (EventSignificance.CRITICAL, EventSignificance.FLASH)]
            if critical:
                report.key_judgments.append(
                    f"{len(critical)} critical narrative(s) identified requiring immediate attention"
                )

        # Body sections
        # Section: Global Overview
        global_content = []
        global_content.append(f"Total Events: {total_events:,}")
        global_content.append(f"Countries Involved: {len(country_events)}")
        global_content.append(f"Conflict Ratio: {100*conflict_events/total_events:.1f}%")
        global_content.append(f"Average Tone: {avg_tone:+.2f}")
        report.body_sections.append(("Global Overview", "\n".join(global_content)))

        # Section: Regional Highlights
        regional_content = []
        for country, count in top_countries[:10]:
            country_evts = [e for e in events if _get(e, 'actor1_country') == country]
            country_conflict = sum(1 for e in country_evts if _get(e, 'quad_class', 0) in (3, 4))
            regional_content.append(
                f"  {country}: {count} events ({100*country_conflict/count:.0f}% conflictual)"
            )
        report.body_sections.append(("Regional Highlights", "\n".join(regional_content)))

        # Section: Watch Items
        if narratives:
            watch_items = []
            for n in narratives[:5]:
                watch_items.append(f"  • [{n.significance.value.upper()}] {n.title}")
            report.body_sections.append(("Watch Items", "\n".join(watch_items)))

        report.source_count = len(set(_get(e, 'source_url') for e in events if _get(e, 'source_url')))
        report.confidence_level = "MODERATE" if report.source_count >= 100 else "LOW"

        self.reports[report.report_id] = report
        return report

    def generate_flash_report(self, event: Any, context: str = "") -> IntelReport:
        """Generate flash report for critical event."""
        report = IntelReport(
            report_id=self._generate_report_id(ReportType.FLASH_REPORT),
            report_type=ReportType.FLASH_REPORT,
            title=f"FLASH: {event.event_description.upper()}",
        )

        report.period_start = event.timestamp
        report.period_end = event.timestamp
        report.event_count = 1

        report.executive_summary = (
            f"IMMEDIATE ATTENTION REQUIRED: {event.actor1_name} has {event.event_description.lower()} "
            f"involving {event.actor2_name}. Event recorded at {event.timestamp.strftime('%Y-%m-%d %H:%M UTC')}. "
            f"Goldstein impact score: {event.goldstein_scale:+.1f}."
        )

        report.key_judgments = [
            f"Primary actor: {event.actor1_name} ({event.actor1_country})",
            f"Target: {event.actor2_name} ({event.actor2_country})",
            f"Event severity: {'CRITICAL' if event.goldstein_scale <= -7 else 'HIGH'}",
        ]

        report.body_sections.append(("Event Details", (
            f"Event Code: {event.event_code}\n"
            f"Quad Class: {event.quad_class}\n"
            f"Goldstein: {event.goldstein_scale:+.1f}\n"
            f"Mentions: {event.num_mentions}\n"
            f"Sources: {event.num_sources}"
        )))

        if context:
            report.body_sections.append(("Context", context))

        report.confidence_level = "HIGH" if event.num_sources >= 3 else "MODERATE"

        self.reports[report.report_id] = report
        return report

    def generate_sitrep(self, events: List[Any], focus_country: str,
                        narratives: List[NarrativeThread] = None) -> IntelReport:
        """Generate situation report for specific country."""
        report = IntelReport(
            report_id=self._generate_report_id(ReportType.SITUATION_REPORT),
            report_type=ReportType.SITUATION_REPORT,
            title=f"SITUATION REPORT: {focus_country}",
        )

        # Filter events
        country_events = [
            e for e in events
            if e.actor1_country == focus_country or e.actor2_country == focus_country
        ]

        if not country_events:
            report.executive_summary = f"No events recorded for {focus_country} in reporting period."
            return report

        timestamps = [e.timestamp for e in country_events]
        report.period_start = min(timestamps)
        report.period_end = max(timestamps)
        report.event_count = len(country_events)

        # Analysis
        as_actor1 = sum(1 for e in country_events if e.actor1_country == focus_country)
        as_actor2 = sum(1 for e in country_events if e.actor2_country == focus_country)
        conflicts = sum(1 for e in country_events if e.quad_class in (3, 4))

        tones = [e.avg_tone for e in country_events if e.avg_tone != 0]
        avg_tone = sum(tones) / len(tones) if tones else 0

        report.executive_summary = (
            f"{focus_country} recorded {len(country_events)} events in the reporting period. "
            f"The country acted as primary actor in {as_actor1} events and was targeted in {as_actor2}. "
            f"Conflict-related events comprised {100*conflicts/len(country_events):.0f}% of total activity. "
            f"Average sentiment: {avg_tone:+.2f}."
        )

        # Key relationships
        partners: Dict[str, int] = defaultdict(int)
        adversaries: Dict[str, int] = defaultdict(int)

        for e in country_events:
            other = e.actor2_country if e.actor1_country == focus_country else e.actor1_country
            if other and other != focus_country:
                if e.quad_class in (1, 2):
                    partners[other] += 1
                elif e.quad_class in (3, 4):
                    adversaries[other] += 1

        report.key_judgments = []

        if partners:
            top_partner = max(partners.items(), key=lambda x: x[1])
            report.key_judgments.append(f"Primary cooperative partner: {top_partner[0]} ({top_partner[1]} events)")

        if adversaries:
            top_adversary = max(adversaries.items(), key=lambda x: x[1])
            report.key_judgments.append(f"Primary conflict relationship: {top_adversary[0]} ({top_adversary[1]} events)")

        # Event breakdown
        event_types: Dict[str, int] = defaultdict(int)
        for e in country_events:
            event_types[e.event_description] += 1

        top_types = sorted(event_types.items(), key=lambda x: -x[1])[:10]
        type_content = "\n".join(f"  {t}: {c}" for t, c in top_types)
        report.body_sections.append(("Event Type Distribution", type_content))

        # Significant events
        significant = sorted(
            [e for e in country_events if e.goldstein_scale <= -5 or e.num_mentions >= 15],
            key=lambda x: x.goldstein_scale
        )[:5]

        if significant:
            sig_content = []
            for e in significant:
                sig_content.append(
                    f"  [{e.timestamp.strftime('%m-%d %H:%M')}] {e.actor1_name} → {e.actor2_name}"
                )
                sig_content.append(f"    {e.event_description} (G: {e.goldstein_scale:+.1f})")
            report.body_sections.append(("Significant Events", "\n".join(sig_content)))

        report.source_count = len(set(e.source_url for e in country_events if e.source_url))
        report.confidence_level = "HIGH" if report.source_count >= 50 else "MODERATE"

        self.reports[report.report_id] = report
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# SHADOWGRAPH: Network Visualization Data
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphNode:
    """A node in the intelligence graph."""
    node_id: str
    label: str
    node_type: str  # country, actor, organization, event
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Visual properties
    size: float = 1.0
    color: str = "#666666"

    # Position (for layout)
    x: float = 0.0
    y: float = 0.0


@dataclass
class GraphEdge:
    """An edge in the intelligence graph."""
    edge_id: str
    source: str
    target: str
    edge_type: str  # cooperative, conflictual, neutral
    weight: float = 1.0

    attributes: Dict[str, Any] = field(default_factory=dict)

    # Visual properties
    color: str = "#999999"
    width: float = 1.0


class SHADOWGRAPH:
    """
    Network visualization and graph analysis system.

    Builds graph representations of intelligence data
    for visualization and network analysis.
    """

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}

    def build_country_network(self, events: List[Any]) -> Dict[str, Any]:
        """Build country interaction network."""
        self.nodes = {}
        self.edges = {}

        # Count interactions
        country_events: Dict[str, int] = defaultdict(int)
        dyad_coop: Dict[str, int] = defaultdict(int)
        dyad_conflict: Dict[str, int] = defaultdict(int)
        dyad_tone: Dict[str, List[float]] = defaultdict(list)

        for e in events:
            if e.actor1_country:
                country_events[e.actor1_country] += 1
            if e.actor2_country:
                country_events[e.actor2_country] += 1

            if e.actor1_country and e.actor2_country and e.actor1_country != e.actor2_country:
                dyad = tuple(sorted([e.actor1_country, e.actor2_country]))
                dyad_key = f"{dyad[0]}-{dyad[1]}"

                if e.quad_class in (1, 2):
                    dyad_coop[dyad_key] += 1
                elif e.quad_class in (3, 4):
                    dyad_conflict[dyad_key] += 1

                if e.avg_tone != 0:
                    dyad_tone[dyad_key].append(e.avg_tone)

        # Create nodes
        max_events = max(country_events.values()) if country_events else 1

        for country, count in country_events.items():
            node = GraphNode(
                node_id=country,
                label=country,
                node_type="country",
                size=0.2 + 0.8 * (count / max_events),
                attributes={"event_count": count},
            )
            self.nodes[country] = node

        # Create edges
        for dyad_key in set(dyad_coop.keys()) | set(dyad_conflict.keys()):
            parts = dyad_key.split("-")
            coop = dyad_coop.get(dyad_key, 0)
            conflict = dyad_conflict.get(dyad_key, 0)
            total = coop + conflict

            if total < 3:
                continue

            tones = dyad_tone.get(dyad_key, [])
            avg_tone = sum(tones) / len(tones) if tones else 0

            # Determine edge type
            if coop > conflict * 2:
                edge_type = "cooperative"
                color = "#00AA00"
            elif conflict > coop * 2:
                edge_type = "conflictual"
                color = "#AA0000"
            else:
                edge_type = "mixed"
                color = "#AAAA00"

            edge = GraphEdge(
                edge_id=dyad_key,
                source=parts[0],
                target=parts[1],
                edge_type=edge_type,
                weight=total,
                color=color,
                width=0.5 + 2 * (total / max(sum(dyad_coop.values()) + sum(dyad_conflict.values()), 1)),
                attributes={
                    "cooperative": coop,
                    "conflictual": conflict,
                    "avg_tone": avg_tone,
                },
            )
            self.edges[dyad_key] = edge

        return self.to_dict()

    def build_actor_network(self, events: List[Any], min_events: int = 5) -> Dict[str, Any]:
        """Build actor interaction network."""
        self.nodes = {}
        self.edges = {}

        actor_events: Dict[str, int] = defaultdict(int)
        actor_pairs: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"coop": 0, "conflict": 0, "tones": []})

        for e in events:
            a1 = e.actor1_name or e.actor1_code
            a2 = e.actor2_name or e.actor2_code

            if a1:
                actor_events[a1] += 1
            if a2:
                actor_events[a2] += 1

            if a1 and a2 and a1 != a2:
                pair = tuple(sorted([a1, a2]))
                pair_key = f"{pair[0]}::{pair[1]}"

                if e.quad_class in (1, 2):
                    actor_pairs[pair_key]["coop"] += 1
                elif e.quad_class in (3, 4):
                    actor_pairs[pair_key]["conflict"] += 1

                if e.avg_tone != 0:
                    actor_pairs[pair_key]["tones"].append(e.avg_tone)

        # Filter to significant actors
        significant_actors = {a for a, c in actor_events.items() if c >= min_events}

        max_events = max((c for a, c in actor_events.items() if a in significant_actors), default=1)

        for actor in significant_actors:
            count = actor_events[actor]
            node = GraphNode(
                node_id=actor,
                label=actor[:20],
                node_type="actor",
                size=0.2 + 0.8 * (count / max_events),
                attributes={"event_count": count},
            )
            self.nodes[actor] = node

        # Create edges between significant actors
        for pair_key, data in actor_pairs.items():
            parts = pair_key.split("::")
            if parts[0] not in significant_actors or parts[1] not in significant_actors:
                continue

            total = data["coop"] + data["conflict"]
            if total < 2:
                continue

            if data["coop"] > data["conflict"]:
                edge_type = "cooperative"
                color = "#00AA00"
            elif data["conflict"] > data["coop"]:
                edge_type = "conflictual"
                color = "#AA0000"
            else:
                edge_type = "mixed"
                color = "#AAAA00"

            edge = GraphEdge(
                edge_id=pair_key,
                source=parts[0],
                target=parts[1],
                edge_type=edge_type,
                weight=total,
                color=color,
                attributes={
                    "cooperative": data["coop"],
                    "conflictual": data["conflict"],
                },
            )
            self.edges[pair_key] = edge

        return self.to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary format."""
        return {
            "nodes": [
                {
                    "id": n.node_id,
                    "label": n.label,
                    "type": n.node_type,
                    "size": n.size,
                    "color": n.color,
                    **n.attributes,
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.edge_id,
                    "source": e.source,
                    "target": e.target,
                    "type": e.edge_type,
                    "weight": e.weight,
                    "color": e.color,
                    **e.attributes,
                }
                for e in self.edges.values()
            ],
            "stats": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            }
        }

    def to_ascii(self, max_nodes: int = 15) -> str:
        """Generate ASCII representation."""
        lines = [
            "┌" + "─" * 78 + "┐",
            "│" + "SHADOWGRAPH NETWORK VISUALIZATION".center(78) + "│",
            "├" + "─" * 78 + "┤",
        ]

        # Top nodes by size
        top_nodes = sorted(self.nodes.values(), key=lambda n: -n.size)[:max_nodes]

        lines.append("│ NODES:" + " " * 71 + "│")
        for node in top_nodes:
            bar_len = int(node.size * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            count = node.attributes.get("event_count", 0)
            lines.append(f"│   {node.label[:15]:15s} [{bar}] {count:5d} │")

        lines.append("├" + "─" * 78 + "┤")
        lines.append("│ EDGES:" + " " * 71 + "│")

        # Top edges by weight
        top_edges = sorted(self.edges.values(), key=lambda e: -e.weight)[:10]

        for edge in top_edges:
            indicator = {"cooperative": "+++", "conflictual": "---", "mixed": "+-+"}.get(edge.edge_type, "???")
            lines.append(f"│   {edge.source[:10]:10s} {indicator} {edge.target[:10]:10s} │ weight: {edge.weight:5.0f} │")

        lines.append("├" + "─" * 78 + "┤")
        lines.append(f"│ Total: {len(self.nodes)} nodes, {len(self.edges)} edges".ljust(78) + "│")
        lines.append("└" + "─" * 78 + "┘")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PRISM-WEAVER: Main Integration System
# ═══════════════════════════════════════════════════════════════════════════════

class PRISM_WEAVER:
    """
    Main intelligence narrative system integrating all components.

    Codename: PRISM-WEAVER

    Capabilities:
    - STORMGLASS: Event correlation
    - LOOKING-GLASS: Target profiling
    - THREADNEEDLE: Narrative extraction
    - GHOSTWRITER: Report generation
    - SHADOWGRAPH: Network visualization
    """

    def __init__(self):
        self.stormglass = STORMGLASS()
        self.looking_glass = LOOKING_GLASS()
        self.threadneedle = THREADNEEDLE()
        self.ghostwriter = GHOSTWRITER()
        self.shadowgraph = SHADOWGRAPH()

        self.initialized = datetime.now(timezone.utc)
        self.events_processed = 0

    def ingest_events(self, events: List[Any]) -> Dict[str, Any]:
        """Ingest events into all subsystems."""
        # Correlate with STORMGLASS
        correlations = self.stormglass.ingest(events)

        self.events_processed += len(events)

        return {
            "events_ingested": len(events),
            "correlations_found": correlations,
            "total_processed": self.events_processed,
        }

    def build_target(self, name: str, events: List[Any]) -> TargetPackage:
        """Build target package."""
        return self.looking_glass.build_target_from_events(name, events)

    def extract_narratives(self, events: List[Any]) -> List[NarrativeThread]:
        """Extract narrative threads."""
        return self.threadneedle.extract_narratives(events)

    def generate_daily_brief(self, events: List[Any]) -> IntelReport:
        """Generate daily intelligence brief."""
        narratives = self.threadneedle.extract_narratives(events)
        return self.ghostwriter.generate_daily_brief(events, narratives)

    def generate_sitrep(self, events: List[Any], country: str) -> IntelReport:
        """Generate situation report for country."""
        return self.ghostwriter.generate_sitrep(events, country)

    def build_network(self, events: List[Any], network_type: str = "country") -> Dict[str, Any]:
        """Build network visualization."""
        if network_type == "country":
            return self.shadowgraph.build_country_network(events)
        else:
            return self.shadowgraph.build_actor_network(events)

    def get_status(self) -> str:
        """Get system status."""
        uptime = datetime.now(timezone.utc) - self.initialized

        return f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                         PRISM-WEAVER STATUS                                  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ System Online: {self.initialized.strftime('%Y-%m-%d %H:%M UTC'):60} ┃
┃ Uptime: {str(uptime).split('.')[0]:67} ┃
┃ Events Processed: {self.events_processed:57,} ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ SUBSYSTEM STATUS                                                             ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ STORMGLASS    │ Selectors: {len(self.stormglass.selectors):5} │ Correlated: {len(self.stormglass.correlated_events):8}      ┃
┃ LOOKING-GLASS │ Targets: {len(self.looking_glass.targets):7} │                              ┃
┃ THREADNEEDLE  │ Threads: {len(self.threadneedle.threads):7} │                              ┃
┃ GHOSTWRITER   │ Reports: {len(self.ghostwriter.reports):7} │                              ┃
┃ SHADOWGRAPH   │ Nodes: {len(self.shadowgraph.nodes):9} │ Edges: {len(self.shadowgraph.edges):9}          ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON ACCESS
# ═══════════════════════════════════════════════════════════════════════════════

_prism_weaver: Optional[PRISM_WEAVER] = None


def get_prism_weaver() -> PRISM_WEAVER:
    """Get global PRISM-WEAVER instance."""
    global _prism_weaver
    if _prism_weaver is None:
        _prism_weaver = PRISM_WEAVER()
    return _prism_weaver


def reset_prism_weaver():
    """Reset PRISM-WEAVER instance."""
    global _prism_weaver
    _prism_weaver = None
