"""Temporal Graph Dynamics for the Meta-Capability Hypergraph.

Enables:
- Time-series state tracking for nodes and edges
- Propagation dynamics through the graph
- Temporal pattern queries and analysis
- Causal inference and dependency tracking
- Predictive dynamics based on historical patterns
"""

from __future__ import annotations

import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from .hypergraph import Hypergraph, HyperNode, HyperEdge, EdgeType


class TemporalResolution(str, Enum):
    """Resolution for temporal snapshots."""
    TICK = "tick"           # Discrete simulation ticks
    MILLISECOND = "ms"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class DynamicsType(str, Enum):
    """Types of temporal dynamics."""
    ACTIVATION = "activation"       # Node becomes active
    DEACTIVATION = "deactivation"   # Node becomes inactive
    PROPAGATION = "propagation"     # Signal propagates through edge
    DECAY = "decay"                 # Value decays over time
    GROWTH = "growth"               # Value grows over time
    OSCILLATION = "oscillation"     # Periodic behavior
    CASCADE = "cascade"             # Triggered chain reaction
    EMERGENCE = "emergence"         # New pattern emerges
    BIFURCATION = "bifurcation"     # System splits into states


class PropagationMode(str, Enum):
    """How signals propagate through edges."""
    IMMEDIATE = "immediate"     # Instant propagation
    DELAYED = "delayed"         # Fixed delay
    WEIGHTED = "weighted"       # Delay based on edge weight
    PROBABILISTIC = "prob"      # Random propagation
    THRESHOLD = "threshold"     # Only if activation exceeds threshold
    INHIBITORY = "inhibitory"   # Suppresses target activation


@dataclass
class TemporalState:
    """State of a node/edge at a specific time."""

    timestamp: datetime
    tick: int = 0

    # Activation and energy
    activation: float = 0.0      # Current activation level [0, 1]
    energy: float = 1.0          # Available energy for propagation
    potential: float = 0.0       # Accumulated potential

    # State flags
    active: bool = True
    locked: bool = False         # Prevent changes
    refractory: bool = False     # In refractory period

    # Values
    values: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    triggered_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "TemporalState":
        """Create a copy of this state."""
        return TemporalState(
            timestamp=self.timestamp,
            tick=self.tick,
            activation=self.activation,
            energy=self.energy,
            potential=self.potential,
            active=self.active,
            locked=self.locked,
            refractory=self.refractory,
            values=dict(self.values),
            triggered_by=self.triggered_by,
            metadata=dict(self.metadata),
        )


@dataclass
class TemporalEvent:
    """An event in the temporal graph."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    event_type: DynamicsType = DynamicsType.ACTIVATION
    timestamp: datetime = field(default_factory=datetime.now)
    tick: int = 0

    # Source and target
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    edge_id: Optional[str] = None

    # Event data
    value: float = 0.0
    delta: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    # Causality
    caused_by: Optional[str] = None  # Event ID that caused this
    causes: List[str] = field(default_factory=list)  # Events this caused

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicsRule:
    """A rule that governs temporal dynamics."""

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    name: str = ""

    # Trigger conditions
    trigger_type: DynamicsType = DynamicsType.ACTIVATION
    trigger_condition: Optional[Callable[[TemporalState, "TemporalHypergraph"], bool]] = None

    # Action
    action: Optional[Callable[[str, TemporalState, "TemporalHypergraph"], TemporalState]] = None

    # Propagation settings
    propagation_mode: PropagationMode = PropagationMode.IMMEDIATE
    propagation_delay: int = 0  # In ticks
    propagation_weight: float = 1.0

    # Constraints
    min_activation: float = 0.0
    max_activation: float = 1.0
    refractory_period: int = 0  # Ticks before can fire again

    # Priority (higher = evaluated first)
    priority: int = 0

    # Active flag
    enabled: bool = True


@dataclass
class TemporalPattern:
    """A pattern detected in temporal dynamics."""

    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    pattern_type: str = ""

    # Involved elements
    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)

    # Temporal bounds
    start_tick: int = 0
    end_tick: int = 0
    duration: int = 0

    # Pattern data
    sequence: List[TemporalEvent] = field(default_factory=list)
    periodicity: Optional[int] = None  # Period in ticks if periodic
    confidence: float = 1.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemporalHypergraph(Hypergraph):
    """Hypergraph with temporal dynamics capabilities.

    Extends the base Hypergraph with:
    - Time-series state tracking
    - Dynamics simulation
    - Temporal queries
    - Pattern detection
    - Causal inference
    """

    def __init__(self, resolution: TemporalResolution = TemporalResolution.TICK):
        super().__init__()

        self.resolution = resolution
        self.current_tick = 0
        self.start_time = datetime.now()

        # Temporal state history
        self._node_states: Dict[str, List[TemporalState]] = defaultdict(list)
        self._edge_states: Dict[str, List[TemporalState]] = defaultdict(list)

        # Current states (for fast access)
        self._current_node_states: Dict[str, TemporalState] = {}
        self._current_edge_states: Dict[str, TemporalState] = {}

        # Events
        self._events: List[TemporalEvent] = []
        self._event_index: Dict[str, TemporalEvent] = {}
        self._pending_events: List[Tuple[int, TemporalEvent]] = []  # (tick, event)

        # Dynamics rules
        self._rules: Dict[str, DynamicsRule] = {}
        self._rules_by_type: Dict[DynamicsType, List[str]] = defaultdict(list)

        # Patterns
        self._detected_patterns: List[TemporalPattern] = []

        # Causal graph
        self._causal_links: Dict[str, Set[str]] = defaultdict(set)  # event_id -> caused events

        # Statistics
        self._temporal_stats = {
            "total_ticks": 0,
            "total_events": 0,
            "propagations": 0,
            "patterns_detected": 0,
            "cascades": 0,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # State Management
    # ─────────────────────────────────────────────────────────────────────────

    def initialize_node_state(
        self,
        node_id: str,
        activation: float = 0.0,
        **values,
    ) -> TemporalState:
        """Initialize temporal state for a node."""
        state = TemporalState(
            timestamp=datetime.now(),
            tick=self.current_tick,
            activation=activation,
            values=values,
        )
        self._node_states[node_id].append(state)
        self._current_node_states[node_id] = state
        return state

    def initialize_edge_state(
        self,
        edge_id: str,
        activation: float = 1.0,
        **values,
    ) -> TemporalState:
        """Initialize temporal state for an edge."""
        state = TemporalState(
            timestamp=datetime.now(),
            tick=self.current_tick,
            activation=activation,
            values=values,
        )
        self._edge_states[edge_id].append(state)
        self._current_edge_states[edge_id] = state
        return state

    def get_node_state(
        self,
        node_id: str,
        tick: Optional[int] = None,
    ) -> Optional[TemporalState]:
        """Get node state at current or specified tick."""
        if tick is None:
            return self._current_node_states.get(node_id)

        history = self._node_states.get(node_id, [])
        for state in reversed(history):
            if state.tick <= tick:
                return state
        return None

    def get_edge_state(
        self,
        edge_id: str,
        tick: Optional[int] = None,
    ) -> Optional[TemporalState]:
        """Get edge state at current or specified tick."""
        if tick is None:
            return self._current_edge_states.get(edge_id)

        history = self._edge_states.get(edge_id, [])
        for state in reversed(history):
            if state.tick <= tick:
                return state
        return None

    def update_node_state(
        self,
        node_id: str,
        activation: Optional[float] = None,
        energy: Optional[float] = None,
        potential: Optional[float] = None,
        active: Optional[bool] = None,
        triggered_by: Optional[str] = None,
        **values,
    ) -> TemporalState:
        """Update node state and record in history."""
        current = self._current_node_states.get(node_id)
        if current is None:
            current = self.initialize_node_state(node_id)

        # Create new state
        new_state = current.copy()
        new_state.timestamp = datetime.now()
        new_state.tick = self.current_tick

        if activation is not None:
            new_state.activation = max(0.0, min(1.0, activation))
        if energy is not None:
            new_state.energy = max(0.0, energy)
        if potential is not None:
            new_state.potential = potential
        if active is not None:
            new_state.active = active
        if triggered_by is not None:
            new_state.triggered_by = triggered_by

        new_state.values.update(values)

        # Record
        self._node_states[node_id].append(new_state)
        self._current_node_states[node_id] = new_state

        return new_state

    def get_state_history(
        self,
        node_id: str,
        start_tick: int = 0,
        end_tick: Optional[int] = None,
    ) -> List[TemporalState]:
        """Get state history for a node."""
        end_tick = end_tick or self.current_tick
        history = self._node_states.get(node_id, [])
        return [s for s in history if start_tick <= s.tick <= end_tick]

    # ─────────────────────────────────────────────────────────────────────────
    # Dynamics Rules
    # ─────────────────────────────────────────────────────────────────────────

    def add_rule(self, rule: DynamicsRule) -> str:
        """Add a dynamics rule."""
        self._rules[rule.rule_id] = rule
        self._rules_by_type[rule.trigger_type].append(rule.rule_id)
        return rule.rule_id

    def create_propagation_rule(
        self,
        name: str,
        edge_type: EdgeType,
        mode: PropagationMode = PropagationMode.WEIGHTED,
        delay: int = 1,
        decay: float = 0.9,
        threshold: float = 0.1,
    ) -> DynamicsRule:
        """Create a standard propagation rule."""

        def condition(state: TemporalState, graph: TemporalHypergraph) -> bool:
            return state.activation >= threshold and not state.refractory

        def action(
            node_id: str,
            state: TemporalState,
            graph: TemporalHypergraph,
        ) -> TemporalState:
            # Find edges from this node
            edges = graph.get_edges_for_node(node_id)

            for edge in edges:
                if edge.edge_type != edge_type:
                    continue

                # Determine targets
                if node_id in edge.source_nodes:
                    targets = edge.target_nodes
                elif edge.bidirectional:
                    targets = [n for n in edge.nodes() if n != node_id]
                else:
                    continue

                # Propagate to targets
                for target_id in targets:
                    signal = state.activation * edge.weight * decay

                    # Schedule propagation event
                    event = TemporalEvent(
                        event_type=DynamicsType.PROPAGATION,
                        tick=graph.current_tick + delay,
                        source_id=node_id,
                        target_id=target_id,
                        edge_id=edge.edge_id,
                        value=signal,
                    )
                    graph.schedule_event(event)

            return state

        rule = DynamicsRule(
            name=name,
            trigger_type=DynamicsType.ACTIVATION,
            trigger_condition=condition,
            action=action,
            propagation_mode=mode,
            propagation_delay=delay,
            min_activation=threshold,
        )

        self.add_rule(rule)
        return rule

    def create_decay_rule(
        self,
        name: str,
        decay_rate: float = 0.05,
        min_activation: float = 0.0,
    ) -> DynamicsRule:
        """Create a decay rule that reduces activation over time."""

        def action(
            node_id: str,
            state: TemporalState,
            graph: TemporalHypergraph,
        ) -> TemporalState:
            new_activation = max(min_activation, state.activation - decay_rate)
            return graph.update_node_state(node_id, activation=new_activation)

        rule = DynamicsRule(
            name=name,
            trigger_type=DynamicsType.DECAY,
            action=action,
            priority=-1,  # Low priority, runs after others
        )

        self.add_rule(rule)
        return rule

    def create_oscillation_rule(
        self,
        name: str,
        period: int,
        amplitude: float = 0.5,
        phase: float = 0.0,
    ) -> DynamicsRule:
        """Create an oscillation rule for periodic behavior."""

        def action(
            node_id: str,
            state: TemporalState,
            graph: TemporalHypergraph,
        ) -> TemporalState:
            t = graph.current_tick
            oscillation = amplitude * math.sin(2 * math.pi * t / period + phase)
            new_activation = 0.5 + oscillation
            return graph.update_node_state(node_id, activation=new_activation)

        rule = DynamicsRule(
            name=name,
            trigger_type=DynamicsType.OSCILLATION,
            action=action,
        )

        self.add_rule(rule)
        return rule

    # ─────────────────────────────────────────────────────────────────────────
    # Event Processing
    # ─────────────────────────────────────────────────────────────────────────

    def schedule_event(self, event: TemporalEvent):
        """Schedule an event for future processing."""
        self._pending_events.append((event.tick, event))
        self._pending_events.sort(key=lambda x: x[0])

    def emit_event(self, event: TemporalEvent):
        """Emit an event immediately."""
        event.timestamp = datetime.now()
        event.tick = self.current_tick

        self._events.append(event)
        self._event_index[event.event_id] = event
        self._temporal_stats["total_events"] += 1

        # Track causality
        if event.caused_by:
            self._causal_links[event.caused_by].add(event.event_id)
            parent = self._event_index.get(event.caused_by)
            if parent:
                parent.causes.append(event.event_id)

        # Process event
        self._process_event(event)

    def _process_event(self, event: TemporalEvent):
        """Process a single event."""
        if event.event_type == DynamicsType.PROPAGATION:
            self._handle_propagation(event)
        elif event.event_type == DynamicsType.ACTIVATION:
            self._handle_activation(event)
        elif event.event_type == DynamicsType.CASCADE:
            self._handle_cascade(event)

    def _handle_propagation(self, event: TemporalEvent):
        """Handle a propagation event."""
        if event.target_id is None:
            return

        current = self.get_node_state(event.target_id)
        if current is None:
            current = self.initialize_node_state(event.target_id)

        # Add incoming signal to potential
        new_potential = current.potential + event.value

        # Check if threshold reached
        threshold = 0.5  # Could be configurable
        if new_potential >= threshold and not current.refractory:
            # Activate node
            self.update_node_state(
                event.target_id,
                activation=min(1.0, new_potential),
                potential=0.0,
                triggered_by=event.source_id,
            )

            # Emit activation event
            activation_event = TemporalEvent(
                event_type=DynamicsType.ACTIVATION,
                source_id=event.target_id,
                value=new_potential,
                caused_by=event.event_id,
            )
            self.emit_event(activation_event)
        else:
            # Just accumulate potential
            self.update_node_state(event.target_id, potential=new_potential)

        self._temporal_stats["propagations"] += 1

    def _handle_activation(self, event: TemporalEvent):
        """Handle an activation event."""
        if event.source_id is None:
            return

        # Apply matching rules
        state = self.get_node_state(event.source_id)
        if state is None:
            return

        rules = self._get_matching_rules(DynamicsType.ACTIVATION)
        for rule in rules:
            if rule.trigger_condition and not rule.trigger_condition(state, self):
                continue
            if rule.action:
                rule.action(event.source_id, state, self)

    def _handle_cascade(self, event: TemporalEvent):
        """Handle a cascade event."""
        self._temporal_stats["cascades"] += 1

        # Find all connected nodes and trigger them
        if event.source_id:
            related = self.find_related(event.source_id, depth=2)
            for node_id in related:
                cascade_event = TemporalEvent(
                    event_type=DynamicsType.ACTIVATION,
                    source_id=node_id,
                    value=event.value * 0.8,  # Decay
                    caused_by=event.event_id,
                )
                self.schedule_event(cascade_event)

    def _get_matching_rules(self, event_type: DynamicsType) -> List[DynamicsRule]:
        """Get rules matching an event type, sorted by priority."""
        rule_ids = self._rules_by_type.get(event_type, [])
        rules = [self._rules[rid] for rid in rule_ids if self._rules[rid].enabled]
        return sorted(rules, key=lambda r: -r.priority)

    # ─────────────────────────────────────────────────────────────────────────
    # Simulation
    # ─────────────────────────────────────────────────────────────────────────

    def tick(self) -> List[TemporalEvent]:
        """Advance simulation by one tick."""
        self.current_tick += 1
        self._temporal_stats["total_ticks"] += 1

        processed_events = []

        # Process pending events for this tick
        while self._pending_events and self._pending_events[0][0] <= self.current_tick:
            _, event = self._pending_events.pop(0)
            self.emit_event(event)
            processed_events.append(event)

        # Apply decay rules to all active nodes
        decay_rules = self._get_matching_rules(DynamicsType.DECAY)
        for node_id, state in self._current_node_states.items():
            if state.activation > 0:
                for rule in decay_rules:
                    if rule.action:
                        rule.action(node_id, state, self)

        # Apply oscillation rules
        osc_rules = self._get_matching_rules(DynamicsType.OSCILLATION)
        for node_id, state in self._current_node_states.items():
            for rule in osc_rules:
                if rule.action:
                    rule.action(node_id, state, self)

        return processed_events

    def run(self, ticks: int) -> Dict[str, Any]:
        """Run simulation for specified number of ticks."""
        all_events = []

        for _ in range(ticks):
            events = self.tick()
            all_events.extend(events)

        return {
            "ticks_run": ticks,
            "events_processed": len(all_events),
            "final_tick": self.current_tick,
            "events": all_events,
        }

    def activate_node(
        self,
        node_id: str,
        activation: float = 1.0,
        propagate: bool = True,
    ) -> TemporalEvent:
        """Manually activate a node."""
        self.update_node_state(node_id, activation=activation)

        event = TemporalEvent(
            event_type=DynamicsType.ACTIVATION,
            source_id=node_id,
            value=activation,
        )

        if propagate:
            self.emit_event(event)
        else:
            self._events.append(event)
            self._event_index[event.event_id] = event

        return event

    def trigger_cascade(
        self,
        node_id: str,
        intensity: float = 1.0,
    ) -> TemporalEvent:
        """Trigger a cascade from a node."""
        event = TemporalEvent(
            event_type=DynamicsType.CASCADE,
            source_id=node_id,
            value=intensity,
        )
        self.emit_event(event)
        return event

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal Queries
    # ─────────────────────────────────────────────────────────────────────────

    def query_at_time(
        self,
        tick: int,
        node_ids: Optional[List[str]] = None,
    ) -> Dict[str, TemporalState]:
        """Query graph state at a specific tick."""
        nodes = node_ids or list(self._node_states.keys())
        result = {}

        for node_id in nodes:
            state = self.get_node_state(node_id, tick)
            if state:
                result[node_id] = state

        return result

    def query_time_range(
        self,
        start_tick: int,
        end_tick: int,
        node_id: Optional[str] = None,
    ) -> Dict[str, List[TemporalState]]:
        """Query state history over a time range."""
        if node_id:
            return {node_id: self.get_state_history(node_id, start_tick, end_tick)}

        result = {}
        for nid in self._node_states.keys():
            history = self.get_state_history(nid, start_tick, end_tick)
            if history:
                result[nid] = history

        return result

    def query_events(
        self,
        event_type: Optional[DynamicsType] = None,
        start_tick: Optional[int] = None,
        end_tick: Optional[int] = None,
        node_id: Optional[str] = None,
    ) -> List[TemporalEvent]:
        """Query events with filters."""
        events = self._events

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if start_tick is not None:
            events = [e for e in events if e.tick >= start_tick]

        if end_tick is not None:
            events = [e for e in events if e.tick <= end_tick]

        if node_id:
            events = [e for e in events if e.source_id == node_id or e.target_id == node_id]

        return events

    def get_activation_timeline(
        self,
        node_id: str,
        start_tick: int = 0,
        end_tick: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """Get activation values over time for a node."""
        history = self.get_state_history(node_id, start_tick, end_tick)
        return [(s.tick, s.activation) for s in history]

    def find_activation_peaks(
        self,
        node_id: str,
        threshold: float = 0.8,
    ) -> List[Tuple[int, float]]:
        """Find peaks in activation for a node."""
        timeline = self.get_activation_timeline(node_id)
        peaks = []

        for i in range(1, len(timeline) - 1):
            prev_val = timeline[i - 1][1]
            curr_val = timeline[i][1]
            next_val = timeline[i + 1][1]

            if curr_val > prev_val and curr_val > next_val and curr_val >= threshold:
                peaks.append(timeline[i])

        return peaks

    # ─────────────────────────────────────────────────────────────────────────
    # Pattern Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_patterns(self) -> List[TemporalPattern]:
        """Detect temporal patterns in the graph dynamics."""
        patterns = []

        # Detect periodic patterns
        patterns.extend(self._detect_periodic_patterns())

        # Detect cascade patterns
        patterns.extend(self._detect_cascade_patterns())

        # Detect emergence patterns
        patterns.extend(self._detect_emergence_patterns())

        self._detected_patterns.extend(patterns)
        self._temporal_stats["patterns_detected"] += len(patterns)

        return patterns

    def _detect_periodic_patterns(self) -> List[TemporalPattern]:
        """Detect periodic activation patterns."""
        patterns = []

        for node_id in self._node_states.keys():
            peaks = self.find_activation_peaks(node_id)

            if len(peaks) < 3:
                continue

            # Check for regular intervals
            intervals = []
            for i in range(1, len(peaks)):
                intervals.append(peaks[i][0] - peaks[i - 1][0])

            if not intervals:
                continue

            avg_interval = sum(intervals) / len(intervals)
            variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)

            # If variance is low, we have a periodic pattern
            if variance < avg_interval * 0.1:
                pattern = TemporalPattern(
                    pattern_type="periodic",
                    node_ids=[node_id],
                    start_tick=peaks[0][0],
                    end_tick=peaks[-1][0],
                    duration=peaks[-1][0] - peaks[0][0],
                    periodicity=int(avg_interval),
                    confidence=1.0 - (variance / avg_interval) if avg_interval > 0 else 0,
                )
                patterns.append(pattern)

        return patterns

    def _detect_cascade_patterns(self) -> List[TemporalPattern]:
        """Detect cascade patterns in event causality."""
        patterns = []

        # Find events that caused many downstream events
        for event_id, caused in self._causal_links.items():
            if len(caused) >= 3:
                event = self._event_index.get(event_id)
                if event:
                    # Trace the cascade
                    cascade_events = self._trace_cascade(event_id)

                    pattern = TemporalPattern(
                        pattern_type="cascade",
                        node_ids=list(set(
                            e.source_id for e in cascade_events if e.source_id
                        )),
                        start_tick=event.tick,
                        end_tick=max(e.tick for e in cascade_events),
                        sequence=cascade_events,
                        metadata={"trigger_event": event_id},
                    )
                    patterns.append(pattern)

        return patterns

    def _trace_cascade(
        self,
        event_id: str,
        max_depth: int = 10,
    ) -> List[TemporalEvent]:
        """Trace all events in a cascade."""
        events = []
        to_trace = [event_id]
        seen = set()
        depth = 0

        while to_trace and depth < max_depth:
            next_trace = []
            for eid in to_trace:
                if eid in seen:
                    continue
                seen.add(eid)

                event = self._event_index.get(eid)
                if event:
                    events.append(event)
                    next_trace.extend(self._causal_links.get(eid, []))

            to_trace = next_trace
            depth += 1

        return events

    def _detect_emergence_patterns(self) -> List[TemporalPattern]:
        """Detect emergence patterns where multiple activations combine."""
        patterns = []

        # Look for nodes that activated after receiving multiple inputs
        for event in self._events:
            if event.event_type != DynamicsType.ACTIVATION:
                continue

            # Find all events that contributed to this activation
            contributing = []
            check_events = [event.caused_by] if event.caused_by else []

            while check_events:
                eid = check_events.pop(0)
                if eid:
                    contributing.append(eid)
                    parent = self._event_index.get(eid)
                    if parent and parent.caused_by:
                        check_events.append(parent.caused_by)

            if len(contributing) >= 3:
                pattern = TemporalPattern(
                    pattern_type="emergence",
                    node_ids=[event.source_id] if event.source_id else [],
                    start_tick=min(
                        self._event_index[e].tick
                        for e in contributing
                        if e in self._event_index
                    ),
                    end_tick=event.tick,
                    metadata={
                        "trigger_count": len(contributing),
                        "result_activation": event.value,
                    },
                )
                patterns.append(pattern)

        return patterns

    # ─────────────────────────────────────────────────────────────────────────
    # Causal Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def get_causal_chain(
        self,
        event_id: str,
        direction: str = "forward",
    ) -> List[TemporalEvent]:
        """Get the causal chain from/to an event."""
        chain = []
        event = self._event_index.get(event_id)

        if not event:
            return chain

        if direction == "forward":
            # Get all effects
            to_check = [event_id]
            seen = set()

            while to_check:
                eid = to_check.pop(0)
                if eid in seen:
                    continue
                seen.add(eid)

                evt = self._event_index.get(eid)
                if evt:
                    chain.append(evt)
                    to_check.extend(self._causal_links.get(eid, []))
        else:
            # Get all causes (backward)
            current = event
            while current:
                chain.append(current)
                if current.caused_by:
                    current = self._event_index.get(current.caused_by)
                else:
                    current = None

        return chain

    def compute_influence(
        self,
        source_id: str,
        target_id: str,
    ) -> float:
        """Compute the temporal influence of source on target."""
        # Count how many times source activation led to target activation
        source_events = self.query_events(
            event_type=DynamicsType.ACTIVATION,
            node_id=source_id,
        )

        influence_count = 0

        for event in source_events:
            # Check if this led to target activation
            chain = self.get_causal_chain(event.event_id, "forward")
            for evt in chain:
                if evt.source_id == target_id:
                    influence_count += 1
                    break

        if not source_events:
            return 0.0

        return influence_count / len(source_events)

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def get_temporal_stats(self) -> Dict[str, Any]:
        """Get temporal dynamics statistics."""
        return {
            **self._temporal_stats,
            "current_tick": self.current_tick,
            "active_nodes": sum(
                1 for s in self._current_node_states.values()
                if s.activation > 0.1
            ),
            "pending_events": len(self._pending_events),
            "rules_count": len(self._rules),
            "patterns_found": len(self._detected_patterns),
        }

    def to_temporal_dict(self) -> Dict[str, Any]:
        """Export temporal graph to dictionary."""
        base = self.to_dict()

        return {
            **base,
            "temporal": {
                "current_tick": self.current_tick,
                "resolution": self.resolution.value,
                "node_states": {
                    nid: {
                        "activation": state.activation,
                        "energy": state.energy,
                        "potential": state.potential,
                        "active": state.active,
                    }
                    for nid, state in self._current_node_states.items()
                },
                "events": [
                    {
                        "id": e.event_id,
                        "type": e.event_type.value,
                        "tick": e.tick,
                        "source": e.source_id,
                        "target": e.target_id,
                        "value": e.value,
                    }
                    for e in self._events[-100:]  # Last 100 events
                ],
                "rules": [
                    {
                        "id": r.rule_id,
                        "name": r.name,
                        "type": r.trigger_type.value,
                        "enabled": r.enabled,
                    }
                    for r in self._rules.values()
                ],
                "patterns": [
                    {
                        "id": p.pattern_id,
                        "type": p.pattern_type,
                        "nodes": p.node_ids,
                        "periodicity": p.periodicity,
                        "confidence": p.confidence,
                    }
                    for p in self._detected_patterns
                ],
                "stats": self._temporal_stats,
            },
        }


# Global temporal hypergraph instance
_temporal_graph: Optional[TemporalHypergraph] = None


def get_temporal_graph() -> TemporalHypergraph:
    """Get or create the global temporal hypergraph."""
    global _temporal_graph
    if _temporal_graph is None:
        _temporal_graph = TemporalHypergraph()
    return _temporal_graph


def reset_temporal_graph() -> TemporalHypergraph:
    """Reset the global temporal hypergraph."""
    global _temporal_graph
    _temporal_graph = TemporalHypergraph()
    return _temporal_graph
