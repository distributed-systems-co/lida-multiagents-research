"""
NIGHTWATCH Complex Event Processing & Anomaly Detection System

Operational codenames:
- NIGHTWATCH: Main CEP orchestration engine
- DEEPWELL: Statistical anomaly detection
- FLASHPOINT: Threshold-based alerting
- CASCADEFLOW: Event stream correlation
- ECHOLOCATE: Pattern echo detection (recurring patterns)
- FAULTLINE: Structural break detection
- QUICKSILVER: Real-time streaming processor

Classification: UNCLASSIFIED // FOR TRAINING PURPOSES ONLY
"""

from __future__ import annotations
import hashlib
import json
import math
import statistics
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Dict, List, Set, Optional, Tuple, Any, Callable,
    Protocol, Type, TypeVar, Union, Iterator, Deque
)
from collections import defaultdict, deque
import logging
import heapq

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    STATISTICAL = "statistical"          # Standard deviation outliers
    BEHAVIORAL = "behavioral"            # Deviation from baseline patterns
    TEMPORAL = "temporal"                # Unusual timing/frequency
    STRUCTURAL = "structural"            # Network/relationship changes
    VOLUME = "volume"                    # Event count spikes/drops
    SENTIMENT = "sentiment"              # Sudden tone shifts
    GEOGRAPHIC = "geographic"            # Unusual location patterns
    ACTOR = "actor"                      # New or unusual actors
    SEQUENCE = "sequence"                # Unexpected event sequences


class SeverityLevel(str, Enum):
    """Anomaly severity levels."""
    INFO = "info"                # Noteworthy but expected
    LOW = "low"                  # Minor deviation
    MEDIUM = "medium"           # Significant deviation
    HIGH = "high"               # Critical deviation
    CRITICAL = "critical"       # Requires immediate attention


class AlertPriority(str, Enum):
    """Alert priority for notification."""
    ROUTINE = "routine"
    ELEVATED = "elevated"
    URGENT = "urgent"
    IMMEDIATE = "immediate"
    FLASH = "flash"


@dataclass
class AnomalyScore:
    """Composite anomaly scoring."""
    z_score: float = 0.0           # Standard deviations from mean
    percentile: float = 0.0        # Position in distribution (0-100)
    deviation_pct: float = 0.0     # Percent deviation from baseline
    confidence: float = 0.0        # Detection confidence (0-1)

    @property
    def composite_score(self) -> float:
        """Calculate composite anomaly score (0-100)."""
        z_contrib = min(abs(self.z_score) * 15, 40)  # Max 40 from z-score
        pct_contrib = (100 - abs(50 - self.percentile)) / 2.5  # Max 20 from percentile
        dev_contrib = min(self.deviation_pct, 40)  # Max 40 from deviation
        return (z_contrib + pct_contrib + dev_contrib) * self.confidence


@dataclass
class DetectedAnomaly:
    """A detected anomaly in the event stream."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: SeverityLevel
    timestamp: datetime
    score: AnomalyScore
    description: str
    affected_entities: List[str] = field(default_factory=list)
    related_events: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "composite_score": self.score.composite_score,
            "z_score": self.score.z_score,
            "description": self.description,
            "affected_entities": self.affected_entities,
            "event_count": len(self.related_events),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPWELL: Statistical Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TimeSeriesPoint:
    """A point in a time series."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StatisticalModel(ABC):
    """Base class for statistical models."""

    @abstractmethod
    def fit(self, data: List[float]) -> None:
        """Fit the model to historical data."""
        pass

    @abstractmethod
    def is_anomaly(self, value: float) -> Tuple[bool, AnomalyScore]:
        """Check if a value is anomalous."""
        pass


class GaussianModel(StatisticalModel):
    """Gaussian/Normal distribution model."""

    def __init__(self, sigma_threshold: float = 3.0):
        self.sigma_threshold = sigma_threshold
        self.mean: float = 0.0
        self.std: float = 1.0
        self.min_val: float = 0.0
        self.max_val: float = 0.0
        self._fitted = False

    def fit(self, data: List[float]) -> None:
        if len(data) < 2:
            return
        self.mean = statistics.mean(data)
        self.std = statistics.stdev(data) if len(data) > 1 else 1.0
        self.min_val = min(data)
        self.max_val = max(data)
        self._fitted = True

    def is_anomaly(self, value: float) -> Tuple[bool, AnomalyScore]:
        if not self._fitted or self.std == 0:
            return False, AnomalyScore()

        z_score = (value - self.mean) / self.std

        # Calculate percentile using error function approximation
        percentile = 50 * (1 + math.erf(z_score / math.sqrt(2)))

        # Deviation from mean as percentage
        deviation_pct = abs(value - self.mean) / max(abs(self.mean), 1) * 100

        # Confidence based on sample stability
        confidence = min(1.0, 0.5 + abs(z_score) / 10)

        score = AnomalyScore(
            z_score=z_score,
            percentile=percentile,
            deviation_pct=min(deviation_pct, 100),
            confidence=confidence
        )

        is_anomalous = abs(z_score) > self.sigma_threshold
        return is_anomalous, score


class ExponentialMovingAverage(StatisticalModel):
    """EMA-based anomaly detection for trending data."""

    def __init__(self, alpha: float = 0.1, threshold_multiplier: float = 2.5):
        self.alpha = alpha
        self.threshold_multiplier = threshold_multiplier
        self.ema: float = 0.0
        self.ema_variance: float = 1.0
        self._fitted = False

    def fit(self, data: List[float]) -> None:
        if not data:
            return

        self.ema = data[0]
        variance_sum = 0.0

        for value in data[1:]:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
            variance_sum += (value - self.ema) ** 2

        self.ema_variance = variance_sum / max(len(data) - 1, 1)
        self._fitted = True

    def update(self, value: float) -> None:
        """Update EMA with new value."""
        if self._fitted:
            prev_ema = self.ema
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
            # Update variance estimate
            self.ema_variance = (self.alpha * (value - prev_ema) ** 2 +
                                (1 - self.alpha) * self.ema_variance)

    def is_anomaly(self, value: float) -> Tuple[bool, AnomalyScore]:
        if not self._fitted:
            return False, AnomalyScore()

        std = math.sqrt(max(self.ema_variance, 1e-10))
        z_score = (value - self.ema) / std

        deviation_pct = abs(value - self.ema) / max(abs(self.ema), 1) * 100
        percentile = 50 * (1 + math.erf(z_score / math.sqrt(2)))

        score = AnomalyScore(
            z_score=z_score,
            percentile=percentile,
            deviation_pct=min(deviation_pct, 100),
            confidence=0.8
        )

        is_anomalous = abs(z_score) > self.threshold_multiplier
        return is_anomalous, score


class DEEPWELL:
    """Statistical anomaly detection system."""

    def __init__(self):
        self.time_series: Dict[str, List[TimeSeriesPoint]] = defaultdict(list)
        self.models: Dict[str, StatisticalModel] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.detected_anomalies: List[DetectedAnomaly] = []
        self._anomaly_counter = 0

    def _generate_anomaly_id(self) -> str:
        self._anomaly_counter += 1
        return f"ANOM-{self._anomaly_counter:06d}"

    def add_series(self, series_name: str, model: Optional[StatisticalModel] = None) -> None:
        """Add a new time series to monitor."""
        if series_name not in self.models:
            self.models[series_name] = model or GaussianModel()

    def record_value(self, series_name: str, value: float,
                     timestamp: Optional[datetime] = None,
                     metadata: Optional[Dict] = None) -> Optional[DetectedAnomaly]:
        """Record a value and check for anomalies."""
        if series_name not in self.models:
            self.add_series(series_name)

        ts = timestamp or datetime.now(timezone.utc)
        point = TimeSeriesPoint(ts, value, metadata or {})
        self.time_series[series_name].append(point)

        # Keep only last 1000 points per series
        if len(self.time_series[series_name]) > 1000:
            self.time_series[series_name] = self.time_series[series_name][-1000:]

        # Fit model if we have enough data
        values = [p.value for p in self.time_series[series_name]]
        if len(values) >= 10:
            self.models[series_name].fit(values[:-1])  # Exclude current for detection

        # Check for anomaly
        is_anomalous, score = self.models[series_name].is_anomaly(value)

        if is_anomalous:
            severity = self._score_to_severity(score)
            anomaly = DetectedAnomaly(
                anomaly_id=self._generate_anomaly_id(),
                anomaly_type=AnomalyType.STATISTICAL,
                severity=severity,
                timestamp=ts,
                score=score,
                description=f"Statistical anomaly in {series_name}: {value:.2f} (z={score.z_score:.2f})",
                affected_entities=[series_name],
                context={"series": series_name, "value": value, **(metadata or {})}
            )
            self.detected_anomalies.append(anomaly)
            return anomaly

        return None

    def _score_to_severity(self, score: AnomalyScore) -> SeverityLevel:
        """Convert anomaly score to severity level."""
        composite = score.composite_score
        if composite >= 80:
            return SeverityLevel.CRITICAL
        elif composite >= 60:
            return SeverityLevel.HIGH
        elif composite >= 40:
            return SeverityLevel.MEDIUM
        elif composite >= 20:
            return SeverityLevel.LOW
        return SeverityLevel.INFO

    def compute_baseline(self, series_name: str,
                        window_hours: int = 24) -> Dict[str, float]:
        """Compute baseline statistics for a series."""
        if series_name not in self.time_series:
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        values = [p.value for p in self.time_series[series_name]
                  if p.timestamp >= cutoff]

        if not values:
            return {}

        baseline = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

        self.baselines[series_name] = baseline
        return baseline

    def get_recent_anomalies(self, hours: int = 24) -> List[DetectedAnomaly]:
        """Get anomalies from the last N hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [a for a in self.detected_anomalies if a.timestamp >= cutoff]


# ═══════════════════════════════════════════════════════════════════════════════
# FLASHPOINT: Threshold-Based Alerting
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThresholdRule:
    """A threshold-based alerting rule."""
    rule_id: str
    name: str
    metric: str
    operator: str  # gt, gte, lt, lte, eq, neq
    threshold: float
    severity: SeverityLevel
    priority: AlertPriority
    cooldown_minutes: int = 5
    description: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """An alert triggered by a threshold rule."""
    alert_id: str
    rule_id: str
    rule_name: str
    timestamp: datetime
    priority: AlertPriority
    severity: SeverityLevel
    metric: str
    current_value: float
    threshold: float
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class FLASHPOINT:
    """Threshold-based alerting system."""

    OPERATORS = {
        "gt": lambda x, t: x > t,
        "gte": lambda x, t: x >= t,
        "lt": lambda x, t: x < t,
        "lte": lambda x, t: x <= t,
        "eq": lambda x, t: x == t,
        "neq": lambda x, t: x != t,
    }

    def __init__(self):
        self.rules: Dict[str, ThresholdRule] = {}
        self.alerts: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self._alert_counter = 0

    def _generate_alert_id(self) -> str:
        self._alert_counter += 1
        return f"ALRT-{self._alert_counter:06d}"

    def add_rule(self, rule: ThresholdRule) -> None:
        """Add a threshold rule."""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a threshold rule."""
        if rule_id in self.rules:
            del self.rules[rule_id]
            return True
        return False

    def check_value(self, metric: str, value: float,
                    context: Optional[Dict] = None) -> List[Alert]:
        """Check a value against all applicable rules."""
        triggered_alerts = []
        now = datetime.now(timezone.utc)

        for rule in self.rules.values():
            if rule.metric != metric:
                continue

            # Check cooldown
            if rule.rule_id in self.last_alert_times:
                cooldown_until = (self.last_alert_times[rule.rule_id] +
                                 timedelta(minutes=rule.cooldown_minutes))
                if now < cooldown_until:
                    continue

            # Check threshold
            op_func = self.OPERATORS.get(rule.operator)
            if op_func and op_func(value, rule.threshold):
                alert = Alert(
                    alert_id=self._generate_alert_id(),
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    timestamp=now,
                    priority=rule.priority,
                    severity=rule.severity,
                    metric=metric,
                    current_value=value,
                    threshold=rule.threshold,
                    message=f"{rule.name}: {metric}={value} {rule.operator} {rule.threshold}",
                    context=context or {}
                )
                self.alerts.append(alert)
                self.last_alert_times[rule.rule_id] = now
                triggered_alerts.append(alert)

        return triggered_alerts

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """Acknowledge an alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now(timezone.utc)
                alert.acknowledged_by = user
                return True
        return False

    def get_active_alerts(self, min_priority: Optional[AlertPriority] = None,
                         unacknowledged_only: bool = True) -> List[Alert]:
        """Get active alerts."""
        priority_order = list(AlertPriority)

        alerts = self.alerts
        if unacknowledged_only:
            alerts = [a for a in alerts if not a.acknowledged]

        if min_priority:
            min_idx = priority_order.index(min_priority)
            alerts = [a for a in alerts
                     if priority_order.index(a.priority) >= min_idx]

        return sorted(alerts,
                     key=lambda a: (priority_order.index(a.priority), a.timestamp),
                     reverse=True)

    def create_standard_rules(self) -> None:
        """Create standard threshold rules for intelligence monitoring."""
        standard_rules = [
            ThresholdRule(
                rule_id="evt-volume-spike",
                name="Event Volume Spike",
                metric="event_count_hourly",
                operator="gt",
                threshold=10000,
                severity=SeverityLevel.HIGH,
                priority=AlertPriority.URGENT,
                description="Hourly event count exceeds threshold"
            ),
            ThresholdRule(
                rule_id="tone-critical-neg",
                name="Critical Negative Tone",
                metric="avg_tone",
                operator="lt",
                threshold=-5.0,
                severity=SeverityLevel.HIGH,
                priority=AlertPriority.ELEVATED,
                description="Average tone drops critically negative"
            ),
            ThresholdRule(
                rule_id="goldstein-conflict",
                name="High Conflict Goldstein",
                metric="avg_goldstein",
                operator="lt",
                threshold=-7.0,
                severity=SeverityLevel.CRITICAL,
                priority=AlertPriority.IMMEDIATE,
                description="Goldstein scale indicates severe conflict"
            ),
            ThresholdRule(
                rule_id="actor-surge",
                name="Actor Event Surge",
                metric="actor_event_count",
                operator="gt",
                threshold=100,
                severity=SeverityLevel.MEDIUM,
                priority=AlertPriority.ELEVATED,
                description="Single actor event count surge"
            ),
        ]

        for rule in standard_rules:
            self.add_rule(rule)


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADEFLOW: Event Stream Correlation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CorrelationRule:
    """Rule for correlating events."""
    rule_id: str
    name: str
    event_patterns: List[Dict[str, Any]]  # Patterns to match
    time_window_minutes: int
    min_matches: int = 2
    correlation_fields: List[str] = field(default_factory=list)  # Fields to correlate on
    output_severity: SeverityLevel = SeverityLevel.MEDIUM


@dataclass
class CorrelatedEventGroup:
    """A group of correlated events."""
    group_id: str
    rule_id: str
    rule_name: str
    timestamp: datetime
    events: List[Dict[str, Any]]
    correlation_key: str
    severity: SeverityLevel
    summary: str


class CASCADEFLOW:
    """Event stream correlation engine."""

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.rules: Dict[str, CorrelationRule] = {}
        self.event_buffer: Deque[Dict[str, Any]] = deque(maxlen=10000)
        self.correlations: List[CorrelatedEventGroup] = []
        self._group_counter = 0

    def _generate_group_id(self) -> str:
        self._group_counter += 1
        return f"CORR-{self._group_counter:06d}"

    def add_rule(self, rule: CorrelationRule) -> None:
        """Add a correlation rule."""
        self.rules[rule.rule_id] = rule

    def _matches_pattern(self, event: Dict[str, Any],
                         pattern: Dict[str, Any]) -> bool:
        """Check if an event matches a pattern."""
        for key, expected in pattern.items():
            if key not in event:
                return False
            actual = event[key]

            if isinstance(expected, dict):
                # Range check
                if "min" in expected and actual < expected["min"]:
                    return False
                if "max" in expected and actual > expected["max"]:
                    return False
                if "contains" in expected and expected["contains"] not in str(actual):
                    return False
                if "in" in expected and actual not in expected["in"]:
                    return False
            elif actual != expected:
                return False

        return True

    def _get_correlation_key(self, event: Dict[str, Any],
                             fields: List[str]) -> str:
        """Generate correlation key from event fields."""
        key_parts = []
        for field in fields:
            if field in event:
                key_parts.append(f"{field}:{event[field]}")
        return "|".join(sorted(key_parts))

    def ingest_event(self, event: Dict[str, Any]) -> List[CorrelatedEventGroup]:
        """Ingest an event and check for correlations."""
        now = datetime.now(timezone.utc)
        event["_ingested_at"] = now
        self.event_buffer.append(event)

        # Clean old events
        cutoff = now - timedelta(minutes=self.window_minutes)
        while self.event_buffer and self.event_buffer[0].get("_ingested_at", now) < cutoff:
            self.event_buffer.popleft()

        triggered_groups = []

        for rule in self.rules.values():
            # Find matching events
            matched_events = []
            for buf_event in self.event_buffer:
                for pattern in rule.event_patterns:
                    if self._matches_pattern(buf_event, pattern):
                        matched_events.append(buf_event)
                        break

            if len(matched_events) < rule.min_matches:
                continue

            # Group by correlation key
            if rule.correlation_fields:
                key_groups: Dict[str, List[Dict]] = defaultdict(list)
                for ev in matched_events:
                    key = self._get_correlation_key(ev, rule.correlation_fields)
                    key_groups[key].append(ev)

                for corr_key, group_events in key_groups.items():
                    if len(group_events) >= rule.min_matches:
                        group = CorrelatedEventGroup(
                            group_id=self._generate_group_id(),
                            rule_id=rule.rule_id,
                            rule_name=rule.name,
                            timestamp=now,
                            events=group_events,
                            correlation_key=corr_key,
                            severity=rule.output_severity,
                            summary=f"{rule.name}: {len(group_events)} events correlated on {corr_key}"
                        )
                        self.correlations.append(group)
                        triggered_groups.append(group)
            else:
                group = CorrelatedEventGroup(
                    group_id=self._generate_group_id(),
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    timestamp=now,
                    events=matched_events,
                    correlation_key="*",
                    severity=rule.output_severity,
                    summary=f"{rule.name}: {len(matched_events)} correlated events"
                )
                self.correlations.append(group)
                triggered_groups.append(group)

        return triggered_groups

    def create_standard_rules(self) -> None:
        """Create standard correlation rules."""
        rules = [
            CorrelationRule(
                rule_id="escalation-sequence",
                name="Escalation Sequence",
                event_patterns=[
                    {"quad_class": {"in": [3, 4]}},  # Conflict events
                ],
                time_window_minutes=120,
                min_matches=5,
                correlation_fields=["actor1_code", "actor2_code"],
                output_severity=SeverityLevel.HIGH
            ),
            CorrelationRule(
                rule_id="multi-country-event",
                name="Multi-Country Event Cascade",
                event_patterns=[
                    {"goldstein": {"max": -5}},
                ],
                time_window_minutes=60,
                min_matches=3,
                correlation_fields=["event_code"],
                output_severity=SeverityLevel.MEDIUM
            ),
            CorrelationRule(
                rule_id="actor-flurry",
                name="Actor Activity Flurry",
                event_patterns=[{}],  # Match all
                time_window_minutes=30,
                min_matches=10,
                correlation_fields=["actor1_code"],
                output_severity=SeverityLevel.MEDIUM
            ),
        ]

        for rule in rules:
            self.add_rule(rule)


# ═══════════════════════════════════════════════════════════════════════════════
# ECHOLOCATE: Pattern Echo Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatternSignature:
    """A signature representing a detected pattern."""
    signature_id: str
    name: str
    feature_vector: List[float]
    created_at: datetime
    last_seen: datetime
    occurrence_count: int = 1
    tags: List[str] = field(default_factory=list)


@dataclass
class PatternEcho:
    """A detected echo of a previous pattern."""
    echo_id: str
    original_signature: str
    current_signature: str
    similarity: float
    timestamp: datetime
    description: str


class ECHOLOCATE:
    """Pattern echo detection - finds recurring patterns."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.signatures: Dict[str, PatternSignature] = {}
        self.echoes: List[PatternEcho] = []
        self._signature_counter = 0
        self._echo_counter = 0

    def _generate_signature_id(self) -> str:
        self._signature_counter += 1
        return f"SIG-{self._signature_counter:06d}"

    def _generate_echo_id(self) -> str:
        self._echo_counter += 1
        return f"ECHO-{self._echo_counter:06d}"

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(v1) != len(v2) or not v1:
            return 0.0

        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a ** 2 for a in v1))
        norm2 = math.sqrt(sum(b ** 2 for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def extract_features(self, events: List[Dict[str, Any]]) -> List[float]:
        """Extract feature vector from a set of events."""
        if not events:
            return [0.0] * 10

        # Feature extraction
        features = []

        # Event count (normalized)
        features.append(min(len(events) / 100, 1.0))

        # Average tone
        tones = [e.get("avg_tone", 0) for e in events if "avg_tone" in e]
        avg_tone = statistics.mean(tones) if tones else 0
        features.append((avg_tone + 10) / 20)  # Normalize -10 to 10 -> 0 to 1

        # Average Goldstein
        goldsteins = [e.get("goldstein", 0) for e in events if "goldstein" in e]
        avg_gold = statistics.mean(goldsteins) if goldsteins else 0
        features.append((avg_gold + 10) / 20)

        # Quad class distribution
        quad_dist = [0.0, 0.0, 0.0, 0.0]
        for e in events:
            qc = e.get("quad_class", 0)
            if 1 <= qc <= 4:
                quad_dist[qc - 1] += 1
        total = sum(quad_dist) or 1
        features.extend([q / total for q in quad_dist])

        # Actor diversity
        actors = set()
        for e in events:
            if "actor1_code" in e:
                actors.add(e["actor1_code"])
            if "actor2_code" in e:
                actors.add(e["actor2_code"])
        features.append(min(len(actors) / 50, 1.0))

        # Country diversity
        countries = set()
        for e in events:
            if "country_code" in e:
                countries.add(e["country_code"])
        features.append(min(len(countries) / 20, 1.0))

        return features

    def register_pattern(self, events: List[Dict[str, Any]],
                        name: str = "",
                        tags: Optional[List[str]] = None) -> PatternSignature:
        """Register a new pattern signature."""
        now = datetime.now(timezone.utc)
        features = self.extract_features(events)

        signature = PatternSignature(
            signature_id=self._generate_signature_id(),
            name=name or f"Pattern-{self._signature_counter}",
            feature_vector=features,
            created_at=now,
            last_seen=now,
            tags=tags or []
        )

        self.signatures[signature.signature_id] = signature
        return signature

    def find_echoes(self, events: List[Dict[str, Any]]) -> List[PatternEcho]:
        """Find echoes of registered patterns in current events."""
        current_features = self.extract_features(events)
        echoes = []
        now = datetime.now(timezone.utc)

        for sig in self.signatures.values():
            similarity = self._cosine_similarity(current_features, sig.feature_vector)

            if similarity >= self.similarity_threshold:
                # Create current signature for reference
                current_sig = self.register_pattern(events, f"Echo-of-{sig.name}")

                echo = PatternEcho(
                    echo_id=self._generate_echo_id(),
                    original_signature=sig.signature_id,
                    current_signature=current_sig.signature_id,
                    similarity=similarity,
                    timestamp=now,
                    description=f"Pattern echo detected: {sig.name} ({similarity:.1%} similar)"
                )
                echoes.append(echo)

                # Update original signature
                sig.last_seen = now
                sig.occurrence_count += 1

        self.echoes.extend(echoes)
        return echoes


# ═══════════════════════════════════════════════════════════════════════════════
# FAULTLINE: Structural Break Detection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralBreak:
    """A detected structural break in the data."""
    break_id: str
    series_name: str
    timestamp: datetime
    before_mean: float
    after_mean: float
    change_magnitude: float
    change_direction: str  # "increase" or "decrease"
    confidence: float
    description: str


class FAULTLINE:
    """Structural break detection system."""

    def __init__(self, min_segment_size: int = 10,
                 significance_threshold: float = 2.0):
        self.min_segment_size = min_segment_size
        self.significance_threshold = significance_threshold
        self.breaks: List[StructuralBreak] = []
        self._break_counter = 0

    def _generate_break_id(self) -> str:
        self._break_counter += 1
        return f"BREAK-{self._break_counter:06d}"

    def detect_break(self, series_name: str,
                    data: List[TimeSeriesPoint]) -> Optional[StructuralBreak]:
        """Detect structural break in time series using CUSUM-like approach."""
        if len(data) < self.min_segment_size * 2:
            return None

        values = [p.value for p in data]
        n = len(values)

        # Calculate global mean and std
        global_mean = statistics.mean(values)
        global_std = statistics.stdev(values) if n > 1 else 1

        if global_std == 0:
            return None

        # Find potential break point using maximum cumulative deviation
        cumsum = 0.0
        max_deviation = 0.0
        break_idx = 0

        for i, v in enumerate(values):
            cumsum += (v - global_mean) / global_std
            if abs(cumsum) > max_deviation:
                max_deviation = abs(cumsum)
                break_idx = i

        # Validate break point
        if break_idx < self.min_segment_size or break_idx > n - self.min_segment_size:
            return None

        before_values = values[:break_idx]
        after_values = values[break_idx:]

        before_mean = statistics.mean(before_values)
        after_mean = statistics.mean(after_values)

        # Calculate significance
        pooled_std = math.sqrt(
            (statistics.variance(before_values) * (len(before_values) - 1) +
             statistics.variance(after_values) * (len(after_values) - 1)) /
            (n - 2)
        ) if n > 2 else global_std

        if pooled_std == 0:
            return None

        t_stat = abs(after_mean - before_mean) / (pooled_std * math.sqrt(1/len(before_values) + 1/len(after_values)))

        if t_stat < self.significance_threshold:
            return None

        change_mag = abs(after_mean - before_mean)
        change_dir = "increase" if after_mean > before_mean else "decrease"
        confidence = min(t_stat / 5, 1.0)  # Normalize t-stat to confidence

        break_point = StructuralBreak(
            break_id=self._generate_break_id(),
            series_name=series_name,
            timestamp=data[break_idx].timestamp,
            before_mean=before_mean,
            after_mean=after_mean,
            change_magnitude=change_mag,
            change_direction=change_dir,
            confidence=confidence,
            description=f"Structural break in {series_name}: {change_dir} of {change_mag:.2f} ({confidence:.0%} confidence)"
        )

        self.breaks.append(break_point)
        return break_point


# ═══════════════════════════════════════════════════════════════════════════════
# QUICKSILVER: Real-Time Streaming Processor
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StreamWindow:
    """A sliding window of events."""
    window_id: str
    start_time: datetime
    end_time: datetime
    events: List[Dict[str, Any]]
    aggregates: Dict[str, float]


class StreamProcessor(ABC):
    """Base class for stream processors."""

    @abstractmethod
    def process(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single event."""
        pass


class AggregationProcessor(StreamProcessor):
    """Aggregates events over a time window."""

    def __init__(self, window_seconds: int = 60,
                 aggregate_fields: List[str] = None):
        self.window_seconds = window_seconds
        self.aggregate_fields = aggregate_fields or ["avg_tone", "goldstein"]
        self.current_window: List[Dict[str, Any]] = []
        self.window_start: Optional[datetime] = None

    def process(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc)

        if self.window_start is None:
            self.window_start = now

        # Check if window expired
        if (now - self.window_start).total_seconds() >= self.window_seconds:
            # Emit aggregate
            result = self._compute_aggregate()
            self.current_window = [event]
            self.window_start = now
            return result

        self.current_window.append(event)
        return None

    def _compute_aggregate(self) -> Dict[str, Any]:
        """Compute aggregates for current window."""
        if not self.current_window:
            return {}

        aggregates = {
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "event_count": len(self.current_window),
        }

        for field in self.aggregate_fields:
            values = [e.get(field) for e in self.current_window if field in e]
            if values:
                aggregates[f"{field}_mean"] = statistics.mean(values)
                aggregates[f"{field}_min"] = min(values)
                aggregates[f"{field}_max"] = max(values)

        return aggregates


class FilterProcessor(StreamProcessor):
    """Filters events based on conditions."""

    def __init__(self, conditions: Dict[str, Any]):
        self.conditions = conditions

    def process(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for key, expected in self.conditions.items():
            if key not in event:
                return None

            actual = event[key]
            if isinstance(expected, dict):
                if "min" in expected and actual < expected["min"]:
                    return None
                if "max" in expected and actual > expected["max"]:
                    return None
                if "in" in expected and actual not in expected["in"]:
                    return None
            elif actual != expected:
                return None

        return event


class QUICKSILVER:
    """Real-time streaming processor."""

    def __init__(self):
        self.processors: List[Tuple[str, StreamProcessor]] = []
        self.output_buffer: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.stats = {
            "events_processed": 0,
            "events_filtered": 0,
            "aggregates_emitted": 0,
        }

    def add_processor(self, name: str, processor: StreamProcessor) -> None:
        """Add a processor to the pipeline."""
        self.processors.append((name, processor))

    def process_event(self, event: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process an event through all processors."""
        self.stats["events_processed"] += 1
        outputs = []
        current_event = event

        for name, processor in self.processors:
            if current_event is None:
                self.stats["events_filtered"] += 1
                break

            result = processor.process(current_event)

            if isinstance(processor, AggregationProcessor) and result:
                result["_processor"] = name
                outputs.append(result)
                self.stats["aggregates_emitted"] += 1
            elif isinstance(processor, FilterProcessor):
                current_event = result
            else:
                current_event = result

        for output in outputs:
            self.output_buffer.append(output)

        return outputs

    def get_recent_outputs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent processor outputs."""
        return list(self.output_buffer)[-count:]


# ═══════════════════════════════════════════════════════════════════════════════
# NIGHTWATCH: Main CEP Orchestration Engine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class NightwatchAlert:
    """Unified alert from NIGHTWATCH system."""
    alert_id: str
    source: str  # Which subsystem generated it
    timestamp: datetime
    priority: AlertPriority
    severity: SeverityLevel
    title: str
    description: str
    related_data: Dict[str, Any] = field(default_factory=dict)


class NIGHTWATCH:
    """
    Main CEP orchestration engine integrating all subsystems.

    Subsystems:
    - DEEPWELL: Statistical anomaly detection
    - FLASHPOINT: Threshold-based alerting
    - CASCADEFLOW: Event stream correlation
    - ECHOLOCATE: Pattern echo detection
    - FAULTLINE: Structural break detection
    - QUICKSILVER: Real-time streaming processor
    """

    def __init__(self):
        self.deepwell = DEEPWELL()
        self.flashpoint = FLASHPOINT()
        self.cascadeflow = CASCADEFLOW()
        self.echolocate = ECHOLOCATE()
        self.faultline = FAULTLINE()
        self.quicksilver = QUICKSILVER()

        self.alerts: List[NightwatchAlert] = []
        self._alert_counter = 0

        # Initialize standard rules
        self.flashpoint.create_standard_rules()
        self.cascadeflow.create_standard_rules()

        # Set up quicksilver pipeline
        self._setup_quicksilver()

    def _generate_alert_id(self) -> str:
        self._alert_counter += 1
        return f"NW-{self._alert_counter:06d}"

    def _setup_quicksilver(self) -> None:
        """Set up the default quicksilver processing pipeline."""
        # Filter for significant events
        self.quicksilver.add_processor(
            "conflict_filter",
            FilterProcessor({"quad_class": {"in": [3, 4]}})
        )

        # Aggregate over 5-minute windows
        self.quicksilver.add_processor(
            "5min_agg",
            AggregationProcessor(window_seconds=300)
        )

    def _convert_to_alert(self, source: str, data: Any,
                         priority: AlertPriority,
                         severity: SeverityLevel,
                         title: str,
                         description: str) -> NightwatchAlert:
        """Convert subsystem output to unified alert."""
        alert = NightwatchAlert(
            alert_id=self._generate_alert_id(),
            source=source,
            timestamp=datetime.now(timezone.utc),
            priority=priority,
            severity=severity,
            title=title,
            description=description,
            related_data=data if isinstance(data, dict) else {"data": str(data)}
        )
        self.alerts.append(alert)
        return alert

    def ingest_event(self, event: Dict[str, Any]) -> List[NightwatchAlert]:
        """Ingest an event through all subsystems."""
        alerts = []

        # DEEPWELL: Statistical monitoring
        if "avg_tone" in event:
            anomaly = self.deepwell.record_value(
                f"tone_{event.get('country_code', 'global')}",
                event["avg_tone"],
                metadata=event
            )
            if anomaly:
                alerts.append(self._convert_to_alert(
                    "DEEPWELL",
                    anomaly.to_dict(),
                    AlertPriority.ELEVATED,
                    anomaly.severity,
                    f"Anomaly: {anomaly.description}",
                    f"Statistical anomaly detected with z-score {anomaly.score.z_score:.2f}"
                ))

        # FLASHPOINT: Threshold checks
        for metric in ["avg_tone", "goldstein", "event_count"]:
            if metric in event:
                fp_alerts = self.flashpoint.check_value(metric, event[metric], event)
                for fp_alert in fp_alerts:
                    alerts.append(self._convert_to_alert(
                        "FLASHPOINT",
                        {"alert": fp_alert.alert_id, "rule": fp_alert.rule_name},
                        fp_alert.priority,
                        fp_alert.severity,
                        fp_alert.rule_name,
                        fp_alert.message
                    ))

        # CASCADEFLOW: Correlation
        corr_groups = self.cascadeflow.ingest_event(event)
        for group in corr_groups:
            alerts.append(self._convert_to_alert(
                "CASCADEFLOW",
                {"group_id": group.group_id, "event_count": len(group.events)},
                AlertPriority.ELEVATED,
                group.severity,
                group.rule_name,
                group.summary
            ))

        # QUICKSILVER: Stream processing
        self.quicksilver.process_event(event)

        return alerts

    def ingest_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ingest a batch of events."""
        all_alerts = []

        for event in events:
            event_alerts = self.ingest_event(event)
            all_alerts.extend(event_alerts)

        # ECHOLOCATE: Pattern echo detection on batch
        echoes = self.echolocate.find_echoes(events)
        for echo in echoes:
            all_alerts.append(self._convert_to_alert(
                "ECHOLOCATE",
                {"echo_id": echo.echo_id, "similarity": echo.similarity},
                AlertPriority.ROUTINE,
                SeverityLevel.INFO,
                "Pattern Echo Detected",
                echo.description
            ))

        return {
            "events_processed": len(events),
            "alerts_generated": len(all_alerts),
            "alerts": [
                {
                    "id": a.alert_id,
                    "source": a.source,
                    "priority": a.priority.value,
                    "severity": a.severity.value,
                    "title": a.title,
                }
                for a in all_alerts
            ],
            "subsystem_stats": {
                "deepwell_anomalies": len(self.deepwell.detected_anomalies),
                "flashpoint_alerts": len(self.flashpoint.alerts),
                "cascadeflow_correlations": len(self.cascadeflow.correlations),
                "echolocate_echoes": len(self.echolocate.echoes),
                "quicksilver_processed": self.quicksilver.stats["events_processed"],
            }
        }

    def analyze_time_series(self, series_name: str,
                           data: List[TimeSeriesPoint]) -> Dict[str, Any]:
        """Analyze a time series for structural breaks."""
        break_point = self.faultline.detect_break(series_name, data)

        if break_point:
            self._convert_to_alert(
                "FAULTLINE",
                {"break_id": break_point.break_id, "confidence": break_point.confidence},
                AlertPriority.URGENT if break_point.confidence > 0.8 else AlertPriority.ELEVATED,
                SeverityLevel.HIGH if break_point.confidence > 0.8 else SeverityLevel.MEDIUM,
                f"Structural Break in {series_name}",
                break_point.description
            )

        baseline = self.deepwell.compute_baseline(series_name)

        return {
            "series": series_name,
            "baseline": baseline,
            "structural_break": break_point.description if break_point else None,
        }

    def register_known_pattern(self, events: List[Dict[str, Any]],
                               name: str,
                               tags: Optional[List[str]] = None) -> str:
        """Register a known pattern for future echo detection."""
        sig = self.echolocate.register_pattern(events, name, tags)
        return sig.signature_id

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alerts_total": len(self.alerts),
            "alerts_by_source": {
                source: len([a for a in self.alerts if a.source == source])
                for source in ["DEEPWELL", "FLASHPOINT", "CASCADEFLOW",
                              "ECHOLOCATE", "FAULTLINE"]
            },
            "alerts_by_severity": {
                sev.value: len([a for a in self.alerts if a.severity == sev])
                for sev in SeverityLevel
            },
            "subsystems": {
                "deepwell": {
                    "series_count": len(self.deepwell.time_series),
                    "anomalies_detected": len(self.deepwell.detected_anomalies),
                },
                "flashpoint": {
                    "rules_active": len(self.flashpoint.rules),
                    "alerts_triggered": len(self.flashpoint.alerts),
                },
                "cascadeflow": {
                    "rules_active": len(self.cascadeflow.rules),
                    "correlations_found": len(self.cascadeflow.correlations),
                    "buffer_size": len(self.cascadeflow.event_buffer),
                },
                "echolocate": {
                    "patterns_registered": len(self.echolocate.signatures),
                    "echoes_detected": len(self.echolocate.echoes),
                },
                "faultline": {
                    "breaks_detected": len(self.faultline.breaks),
                },
                "quicksilver": self.quicksilver.stats,
            }
        }

    def get_alerts(self, min_priority: Optional[AlertPriority] = None,
                  min_severity: Optional[SeverityLevel] = None,
                  source: Optional[str] = None,
                  hours: int = 24) -> List[NightwatchAlert]:
        """Get filtered alerts."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        alerts = [a for a in self.alerts if a.timestamp >= cutoff]

        if source:
            alerts = [a for a in alerts if a.source == source]

        priority_order = list(AlertPriority)
        severity_order = list(SeverityLevel)

        if min_priority:
            min_p_idx = priority_order.index(min_priority)
            alerts = [a for a in alerts
                     if priority_order.index(a.priority) >= min_p_idx]

        if min_severity:
            min_s_idx = severity_order.index(min_severity)
            alerts = [a for a in alerts
                     if severity_order.index(a.severity) >= min_s_idx]

        return sorted(alerts,
                     key=lambda a: (priority_order.index(a.priority), a.timestamp),
                     reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_nightwatch_instance: Optional[NIGHTWATCH] = None


def get_nightwatch() -> NIGHTWATCH:
    """Get global NIGHTWATCH instance."""
    global _nightwatch_instance
    if _nightwatch_instance is None:
        _nightwatch_instance = NIGHTWATCH()
    return _nightwatch_instance


def reset_nightwatch() -> None:
    """Reset global NIGHTWATCH instance."""
    global _nightwatch_instance
    _nightwatch_instance = None
