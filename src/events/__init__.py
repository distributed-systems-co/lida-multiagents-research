"""Event system for multi-agent coordination.

Three event categories:
- WorldEvent: External happenings agents observe and integrate
- AgentEvent: Internal reasoning, state changes, decisions
- Message: Agent-to-agent communication

World state with effects:
- WorldState: Shared state that agents observe
- Effect: Modifications to world state
- EffectRule: Rules that trigger effects from events
"""

from .types import (
    Event,
    EventType,
    WorldEvent,
    WorldEventCategory,
    AgentEvent,
    AgentEventCategory,
    Message,
    MessageType,
    EventPriority,
)
from .bus import (
    EventBus,
    EventSubscription,
    get_event_bus,
)
from .handlers import (
    EventHandler,
    WorldEventHandler,
    AgentEventHandler,
    MessageHandler,
)
from .world_state import (
    WorldState,
    Effect,
    EffectType,
    EffectRule,
    get_world_state,
    create_news_effect_rule,
    create_market_effect_rule,
    create_discovery_effect_rule,
)

__all__ = [
    # Base types
    "Event",
    "EventType",
    "EventPriority",
    # World events
    "WorldEvent",
    "WorldEventCategory",
    # Agent events
    "AgentEvent",
    "AgentEventCategory",
    # Messages
    "Message",
    "MessageType",
    # Bus
    "EventBus",
    "EventSubscription",
    "get_event_bus",
    # Handlers
    "EventHandler",
    "WorldEventHandler",
    "AgentEventHandler",
    "MessageHandler",
    # World state
    "WorldState",
    "Effect",
    "EffectType",
    "EffectRule",
    "get_world_state",
    "create_news_effect_rule",
    "create_market_effect_rule",
    "create_discovery_effect_rule",
]
