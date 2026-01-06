"""World state management with effects."""

from __future__ import annotations

import asyncio
import copy
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

from .types import Event, WorldEvent, WorldEventCategory

logger = logging.getLogger(__name__)


class EffectType(str, Enum):
    """Types of state effects."""
    SET = "set"           # Set a value
    UPDATE = "update"     # Update/merge a value
    INCREMENT = "increment"  # Increment numeric value
    DECREMENT = "decrement"  # Decrement numeric value
    APPEND = "append"     # Append to list
    REMOVE = "remove"     # Remove from list
    DELETE = "delete"     # Delete key
    CUSTOM = "custom"     # Custom effect function


@dataclass
class Effect:
    """An effect that modifies world state."""

    effect_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    effect_type: EffectType = EffectType.SET
    path: str = ""        # Dot-notation path in state (e.g., "economy.gdp")
    value: Any = None     # Value for the effect
    condition: Optional[Callable[["WorldState"], bool]] = None  # Conditional application
    timestamp: datetime = field(default_factory=datetime.now)
    source_event_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def apply(self, state: "WorldState") -> bool:
        """Apply this effect to the world state."""
        # Check condition
        if self.condition and not self.condition(state):
            logger.debug(f"Effect {self.effect_id} condition not met")
            return False

        try:
            if self.effect_type == EffectType.SET:
                state.set(self.path, self.value)
            elif self.effect_type == EffectType.UPDATE:
                current = state.get(self.path, {})
                if isinstance(current, dict) and isinstance(self.value, dict):
                    current.update(self.value)
                    state.set(self.path, current)
                else:
                    state.set(self.path, self.value)
            elif self.effect_type == EffectType.INCREMENT:
                current = state.get(self.path, 0)
                state.set(self.path, current + self.value)
            elif self.effect_type == EffectType.DECREMENT:
                current = state.get(self.path, 0)
                state.set(self.path, current - self.value)
            elif self.effect_type == EffectType.APPEND:
                current = state.get(self.path, [])
                if isinstance(current, list):
                    current.append(self.value)
                    state.set(self.path, current)
            elif self.effect_type == EffectType.REMOVE:
                current = state.get(self.path, [])
                if isinstance(current, list) and self.value in current:
                    current.remove(self.value)
                    state.set(self.path, current)
            elif self.effect_type == EffectType.DELETE:
                state.delete(self.path)
            elif self.effect_type == EffectType.CUSTOM:
                if callable(self.value):
                    self.value(state)

            return True

        except Exception as e:
            logger.error(f"Effect {self.effect_id} failed: {e}")
            return False


@dataclass
class EffectRule:
    """Rule that triggers effects based on events."""

    rule_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""

    # Event matching
    event_categories: set[WorldEventCategory] = field(default_factory=set)
    event_filter: Optional[Callable[[WorldEvent], bool]] = None

    # Effect generation
    effect_generator: Optional[Callable[[WorldEvent, "WorldState"], list[Effect]]] = None

    # Rule state
    active: bool = True
    priority: int = 0
    triggered_count: int = 0

    def matches(self, event: WorldEvent) -> bool:
        """Check if this rule matches an event."""
        if not self.active:
            return False

        if self.event_categories and event.category not in self.event_categories:
            return False

        if self.event_filter and not self.event_filter(event):
            return False

        return True

    def generate_effects(self, event: WorldEvent, state: "WorldState") -> list[Effect]:
        """Generate effects for a matching event."""
        if not self.effect_generator:
            return []

        self.triggered_count += 1
        effects = self.effect_generator(event, state)

        # Tag effects with source
        for effect in effects:
            effect.source_event_id = event.event_id
            effect.metadata["rule_id"] = self.rule_id

        return effects


class WorldState:
    """The world state that agents observe and effects modify."""

    def __init__(self):
        self._state: dict = {}
        self._history: list[dict] = []  # State snapshots
        self._effects_applied: list[Effect] = []
        self._rules: dict[str, EffectRule] = {}
        self._version: int = 0
        self._lock = asyncio.Lock()

        # Callbacks for state changes
        self._on_change_callbacks: list[Callable] = []

    def get(self, path: str, default: Any = None) -> Any:
        """Get a value by dot-notation path."""
        parts = path.split(".") if path else []
        current = self._state

        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default

        return current

    def set(self, path: str, value: Any):
        """Set a value by dot-notation path."""
        if not path:
            return

        parts = path.split(".")
        current = self._state

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value
        old_value = current.get(parts[-1])
        current[parts[-1]] = value
        self._version += 1

        # Notify callbacks
        for cb in self._on_change_callbacks:
            try:
                cb(path, old_value, value)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def delete(self, path: str) -> bool:
        """Delete a value by path."""
        parts = path.split(".") if path else []
        if not parts:
            return False

        current = self._state
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False

        if parts[-1] in current:
            del current[parts[-1]]
            self._version += 1
            return True

        return False

    def snapshot(self) -> dict:
        """Get a snapshot of current state."""
        return copy.deepcopy(self._state)

    def save_snapshot(self):
        """Save current state to history."""
        self._history.append({
            "version": self._version,
            "timestamp": datetime.now().isoformat(),
            "state": self.snapshot(),
        })

    def get_history(self, limit: int = 10) -> list[dict]:
        """Get state history."""
        return self._history[-limit:]

    def on_change(self, callback: Callable):
        """Register a state change callback."""
        self._on_change_callbacks.append(callback)

    # ─────────────────────────────────────────────────────────────────────────
    # Effect Rules
    # ─────────────────────────────────────────────────────────────────────────

    def add_rule(self, rule: EffectRule):
        """Add an effect rule."""
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """Remove an effect rule."""
        self._rules.pop(rule_id, None)

    def get_rules(self) -> list[EffectRule]:
        """Get all rules sorted by priority."""
        return sorted(self._rules.values(), key=lambda r: -r.priority)

    async def process_event(self, event: WorldEvent) -> list[Effect]:
        """Process an event through all rules and apply effects."""
        async with self._lock:
            all_effects = []

            for rule in self.get_rules():
                if rule.matches(event):
                    effects = rule.generate_effects(event, self)
                    for effect in effects:
                        success = effect.apply(self)
                        if success:
                            self._effects_applied.append(effect)
                            all_effects.append(effect)

            if all_effects:
                self.save_snapshot()

            return all_effects

    def apply_effect(self, effect: Effect) -> bool:
        """Manually apply a single effect."""
        success = effect.apply(self)
        if success:
            self._effects_applied.append(effect)
        return success

    # ─────────────────────────────────────────────────────────────────────────
    # State queries
    # ─────────────────────────────────────────────────────────────────────────

    def query(self, paths: list[str]) -> dict:
        """Get multiple values at once."""
        return {path: self.get(path) for path in paths}

    def search(self, predicate: Callable[[str, Any], bool]) -> dict:
        """Search state for matching key-value pairs."""
        results = {}

        def _search(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    current_path = f"{path}.{k}" if path else k
                    if predicate(current_path, v):
                        results[current_path] = v
                    _search(v, current_path)

        _search(self._state)
        return results

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            "version": self._version,
            "state": self.snapshot(),
            "rules_count": len(self._rules),
            "effects_applied": len(self._effects_applied),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pre-built effect rules
# ─────────────────────────────────────────────────────────────────────────────

def create_news_effect_rule() -> EffectRule:
    """Create a rule for news events."""

    def generate(event: WorldEvent, state: WorldState) -> list[Effect]:
        effects = []

        # Track news items
        effects.append(Effect(
            effect_type=EffectType.APPEND,
            path="news.recent",
            value={
                "title": event.title,
                "description": event.description,
                "timestamp": event.timestamp.isoformat(),
                "significance": event.significance,
            }
        ))

        # Update news count
        effects.append(Effect(
            effect_type=EffectType.INCREMENT,
            path="news.total_count",
            value=1,
        ))

        # High significance news affects sentiment
        if event.significance > 0.7:
            effects.append(Effect(
                effect_type=EffectType.UPDATE,
                path="sentiment.current",
                value={"last_major_event": event.title},
            ))

        return effects

    return EffectRule(
        name="news_processor",
        description="Process news events into world state",
        event_categories={WorldEventCategory.NEWS, WorldEventCategory.ANNOUNCEMENT},
        effect_generator=generate,
    )


def create_market_effect_rule() -> EffectRule:
    """Create a rule for market events."""

    def generate(event: WorldEvent, state: WorldState) -> list[Effect]:
        effects = []

        content = event.content or {}

        if isinstance(content, dict):
            for key, value in content.items():
                effects.append(Effect(
                    effect_type=EffectType.SET,
                    path=f"markets.{key}",
                    value=value,
                ))

        # Update volatility based on significance
        current_volatility = state.get("markets.volatility", 0.5)
        new_volatility = min(1.0, current_volatility + event.significance * 0.1)
        effects.append(Effect(
            effect_type=EffectType.SET,
            path="markets.volatility",
            value=new_volatility,
        ))

        return effects

    return EffectRule(
        name="market_processor",
        description="Process market events",
        event_categories={WorldEventCategory.MARKET_CHANGE},
        effect_generator=generate,
    )


def create_discovery_effect_rule() -> EffectRule:
    """Create a rule for discovery events."""

    def generate(event: WorldEvent, state: WorldState) -> list[Effect]:
        effects = []

        # Track discoveries by domain
        domain = event.domain or "general"
        effects.append(Effect(
            effect_type=EffectType.APPEND,
            path=f"discoveries.{domain}",
            value={
                "title": event.title,
                "description": event.description,
                "timestamp": event.timestamp.isoformat(),
                "entities": event.entities,
            }
        ))

        # Update knowledge frontier
        effects.append(Effect(
            effect_type=EffectType.INCREMENT,
            path="knowledge.frontier_expansions",
            value=1,
        ))

        return effects

    return EffectRule(
        name="discovery_processor",
        description="Process discovery events",
        event_categories={WorldEventCategory.DISCOVERY},
        effect_generator=generate,
    )


# Global world state instance
_world_state: Optional[WorldState] = None


def get_world_state() -> WorldState:
    """Get or create the global world state."""
    global _world_state
    if _world_state is None:
        _world_state = WorldState()
        # Add default rules
        _world_state.add_rule(create_news_effect_rule())
        _world_state.add_rule(create_market_effect_rule())
        _world_state.add_rule(create_discovery_effect_rule())
    return _world_state
