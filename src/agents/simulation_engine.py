"""Multi-Agent Policy Simulation Engine for Post-AGI Wargaming.

Supports three simulation modes:
1. Turn-Based - Discrete events, crisis response, structured decision-making
2. Continuous - Long-arc scenarios, drift modeling, asynchronous actions
3. Negotiation - Bilateral/multilateral deals, ultimatums, bargaining

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      SIMULATION ENGINE                          │
    │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
    │  │  World State   │  │  Event System  │  │  History Tracker │  │
    │  │  (geopolitics, │  │  (actions,     │  │  (counterfactual │  │
    │  │   compute,     │  │   reactions,   │  │   branching,     │  │
    │  │   alliances)   │  │   cascades)    │  │   replay)        │  │
    │  └───────┬────────┘  └───────┬────────┘  └────────┬─────────┘  │
    │          │                   │                    │            │
    │          └───────────────────┼────────────────────┘            │
    │                              │                                 │
    │  ┌───────────────────────────┼───────────────────────────┐     │
    │  │               AGENT ORCHESTRATOR                       │     │
    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐      │     │
    │  │  │ Agent A │ │ Agent B │ │ Agent C │ │ Agent N │      │     │
    │  │  │ (Xi)    │ │(Raimondo│ │(Altman) │ │  ...    │      │     │
    │  │  │         │ │)        │ │         │ │         │      │     │
    │  │  │ Persona │ │ Persona │ │ Persona │ │ Persona │      │     │
    │  │  │ +State  │ │ +State  │ │ +State  │ │ +State  │      │     │
    │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘      │     │
    │  └───────────────────────────────────────────────────────┘     │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Core Types and Enums
# =============================================================================


class SimulationMode(str, Enum):
    """Simulation execution modes."""
    TURN_BASED = "turn_based"
    CONTINUOUS = "continuous"
    NEGOTIATION = "negotiation"


class ActionType(str, Enum):
    """Types of actions agents can take."""
    # Diplomatic
    STATEMENT = "statement"
    PROPOSAL = "proposal"
    THREAT = "threat"
    CONCESSION = "concession"
    ULTIMATUM = "ultimatum"
    ALLIANCE = "alliance"
    SANCTION = "sanction"

    # Policy
    EXECUTIVE_ORDER = "executive_order"
    LEGISLATION = "legislation"
    REGULATION = "regulation"
    TREATY = "treaty"

    # Economic
    TARIFF = "tariff"
    SUBSIDY = "subsidy"
    INVESTMENT = "investment"
    EMBARGO = "embargo"

    # Technology
    EXPORT_CONTROL = "export_control"
    RESEARCH_INITIATIVE = "research_initiative"
    DEPLOYMENT_DECISION = "deployment_decision"
    SAFETY_PAUSE = "safety_pause"

    # Military
    MILITARY_POSTURE = "military_posture"
    EXERCISE = "exercise"
    MOBILIZATION = "mobilization"

    # Internal
    THINK = "think"
    WAIT = "wait"
    ESCALATE = "escalate"
    DEESCALATE = "deescalate"


class EmotionalState(str, Enum):
    """Emotional states affecting decision-making."""
    CALM = "calm"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"
    CONFIDENT = "confident"
    DESPERATE = "desperate"
    TRIUMPHANT = "triumphant"
    HUMILIATED = "humiliated"


class RelationshipType(str, Enum):
    """Types of relationships between agents."""
    ALLY = "ally"
    PARTNER = "partner"
    NEUTRAL = "neutral"
    RIVAL = "rival"
    ADVERSARY = "adversary"
    ENEMY = "enemy"


class EscalationLevel(int, Enum):
    """Escalation ladder levels."""
    PEACE = 0
    TENSION = 1
    CRISIS = 2
    CONFRONTATION = 3
    LIMITED_CONFLICT = 4
    MAJOR_CONFLICT = 5
    TOTAL_WAR = 6


# =============================================================================
# World State
# =============================================================================


@dataclass
class ComputeStock:
    """Global compute resource tracking."""
    total_flops: float  # Exaflops
    ai_training_allocation: float  # Percentage
    military_allocation: float
    civilian_allocation: float
    growth_rate: float  # Annual

    # Geographic distribution
    us_share: float
    china_share: float
    eu_share: float
    other_share: float

    # Chip production
    advanced_node_capacity: float  # 3nm equivalent wafers/month
    tsmc_share: float
    samsung_share: float
    intel_share: float


@dataclass
class AICapabilities:
    """Global AI capability tracking."""
    frontier_model_capability: float = 70.0  # 0-100 scale
    agi_proximity: float = 0.3  # 0-1, estimate
    asi_risk: float = 0.1  # 0-1

    # Lab standings
    lab_capabilities: Dict[str, float] = field(default_factory=dict)

    # Safety metrics
    alignment_confidence: float = 0.5
    containment_effectiveness: float = 0.7

    # Deployment state
    deployed_systems: List[str] = field(default_factory=list)
    autonomous_weapons: bool = False


@dataclass
class GeopoliticalState:
    """Global geopolitical context."""
    escalation_level: EscalationLevel = EscalationLevel.TENSION
    active_conflicts: List[str] = field(default_factory=list)
    active_negotiations: List[str] = field(default_factory=list)
    recent_events: List[str] = field(default_factory=list)

    # Alliance blocs
    alliances: Dict[str, Set[str]] = field(default_factory=dict)

    # Sanctions/embargoes
    sanctions: Dict[str, Set[str]] = field(default_factory=dict)

    # Treaties in effect
    treaties: List[str] = field(default_factory=list)


@dataclass
class WorldState:
    """Complete world state for simulation."""
    timestamp: float = field(default_factory=time.time)
    tick: int = 0

    # Core state components
    compute: ComputeStock = field(default_factory=lambda: ComputeStock(
        total_flops=100.0, ai_training_allocation=0.3, military_allocation=0.1,
        civilian_allocation=0.6, growth_rate=0.35, us_share=0.35, china_share=0.25,
        eu_share=0.15, other_share=0.25, advanced_node_capacity=100000,
        tsmc_share=0.55, samsung_share=0.2, intel_share=0.15
    ))
    ai: AICapabilities = field(default_factory=lambda: AICapabilities())
    geopolitics: GeopoliticalState = field(default_factory=lambda: GeopoliticalState())

    # Custom state variables
    variables: Dict[str, Any] = field(default_factory=dict)

    # State history for rollback
    _history: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of current state."""
        return {
            "timestamp": self.timestamp,
            "tick": self.tick,
            "escalation": self.geopolitics.escalation_level.value,
            "ai_capability": self.ai.frontier_model_capability,
            "agi_proximity": self.ai.agi_proximity,
            "variables": self.variables.copy(),
        }

    def checkpoint(self):
        """Save current state to history."""
        self._history.append(self.snapshot())

    def rollback(self, steps: int = 1):
        """Rollback to previous state."""
        if len(self._history) >= steps:
            target = self._history[-steps]
            self.tick = target["tick"]
            self.variables = target["variables"]

    def advance_tick(self, delta: float = 1.0):
        """Advance simulation time."""
        self.tick += 1
        self.timestamp += delta

        # Natural evolution of state
        self._evolve_compute(delta)
        self._evolve_ai(delta)

    def _evolve_compute(self, delta: float):
        """Natural compute growth."""
        annual_factor = delta / 365.0
        self.compute.total_flops *= (1 + self.compute.growth_rate * annual_factor)

    def _evolve_ai(self, delta: float):
        """Natural AI capability progression."""
        # Capability creep
        capability_growth = 0.02 * delta  # ~2% per tick
        self.ai.frontier_model_capability = min(
            100.0,
            self.ai.frontier_model_capability + capability_growth
        )

        # AGI proximity increases with capability
        if self.ai.frontier_model_capability > 80:
            self.ai.agi_proximity = min(1.0, self.ai.agi_proximity + 0.01 * delta)


# =============================================================================
# Agent State and Persona
# =============================================================================


@dataclass
class AgentState:
    """Dynamic state of an agent during simulation."""
    emotional_state: EmotionalState = EmotionalState.CALM
    stress_level: float = 0.0  # 0-1
    credibility: float = 1.0  # 0-1
    political_capital: float = 1.0  # 0-1

    # Relationship modifiers
    relationship_modifiers: Dict[str, float] = field(default_factory=dict)

    # Recent actions and their effects
    recent_actions: List[Dict[str, Any]] = field(default_factory=list)

    # Commitments and red lines
    active_commitments: List[str] = field(default_factory=list)
    stated_red_lines: List[str] = field(default_factory=list)

    # Private knowledge/beliefs
    private_beliefs: Dict[str, Any] = field(default_factory=dict)

    # Cooldowns (action type -> ticks until available)
    cooldowns: Dict[ActionType, int] = field(default_factory=dict)

    def can_act(self, action_type: ActionType) -> bool:
        """Check if agent can take an action."""
        return self.cooldowns.get(action_type, 0) <= 0

    def apply_cooldown(self, action_type: ActionType, ticks: int):
        """Apply a cooldown to an action type."""
        self.cooldowns[action_type] = ticks

    def tick_cooldowns(self):
        """Reduce all cooldowns by 1."""
        for action_type in list(self.cooldowns.keys()):
            self.cooldowns[action_type] = max(0, self.cooldowns[action_type] - 1)
            if self.cooldowns[action_type] == 0:
                del self.cooldowns[action_type]


@dataclass
class Persona:
    """Agent persona loaded from JSON."""
    name: str
    current_role: str
    category: str

    # Background
    background: Dict[str, Any] = field(default_factory=dict)

    # Personality
    core_traits: List[Dict[str, str]] = field(default_factory=list)
    quirks: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    insecurities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)

    # Communication style
    communication: Dict[str, Any] = field(default_factory=dict)

    # Relationships
    relationships: Dict[str, Any] = field(default_factory=dict)

    # Worldview
    worldview: Dict[str, Any] = field(default_factory=dict)

    # Simulation guide
    simulation_guide: Dict[str, Any] = field(default_factory=dict)

    # Gossip/intel
    gossip: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: Path) -> "Persona":
        """Load persona from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        # Try name, subject, or derive from filename
        name = data.get("name") or data.get("subject")
        if not name:
            # Extract from filename: jensen_huang.json -> Jensen Huang
            name = path.stem.replace("_", " ").title()

        # Extract current_role from multiple possible locations
        background = data.get("background", {})
        current_role = (
            data.get("current_role")
            or background.get("current_position")
            or background.get("title")
            or background.get("role")
            or ""
        )

        # Extract category from multiple possible locations
        category = (
            data.get("category")
            or data.get("domain")
            or background.get("domain")
            or ""
        )

        return cls(
            name=name,
            current_role=current_role,
            category=category,
            background=background,
            core_traits=data.get("personality", {}).get("core_traits", []),
            quirks=data.get("personality", {}).get("quirks", []),
            triggers=data.get("personality", {}).get("triggers", []),
            insecurities=data.get("personality", {}).get("insecurities", []),
            strengths=data.get("personality", {}).get("strengths", []),
            communication=data.get("communication", {}),
            relationships=data.get("relationships", {}),
            worldview=data.get("worldview", {}),
            simulation_guide=data.get("simulation_guide", {}),
            gossip=data.get("_gossip", {}),
        )

    def get_hot_buttons(self) -> List[str]:
        """Get things that provoke this persona."""
        guide = self.simulation_guide
        return guide.get("hot_buttons", []) + self.triggers

    def get_never_say(self) -> List[str]:
        """Get things this persona would never say."""
        return self.simulation_guide.get("never_say", [])

    def get_how_to_flatter(self) -> List[str]:
        """Get ways to flatter this persona."""
        return self.simulation_guide.get("how_to_flatter", [])

    def get_how_to_provoke(self) -> List[str]:
        """Get ways to provoke this persona."""
        return self.simulation_guide.get("how_to_provoke", [])

    def build_system_prompt(self, state: AgentState, world: WorldState) -> str:
        """Build the system prompt for this persona."""
        guide = self.simulation_guide

        prompt_parts = [
            f"You are {self.name}, {self.current_role}.",
            "",
            "## How to Embody This Character",
            guide.get("how_to_embody", "Act naturally as this character."),
            "",
            "## Voice Characteristics",
        ]

        voice = guide.get("voice_characteristics", [])
        if isinstance(voice, list):
            for v in voice:
                prompt_parts.append(f"- {v}")
        elif isinstance(voice, dict):
            for k, v in voice.items():
                prompt_parts.append(f"- {k}: {v}")

        prompt_parts.extend([
            "",
            "## Things You Would Never Say",
        ])
        for ns in self.get_never_say():
            prompt_parts.append(f"- \"{ns}\"")

        prompt_parts.extend([
            "",
            "## Your Core Beliefs",
        ])
        beliefs = self.worldview.get("core_beliefs", [])
        for belief in beliefs:
            prompt_parts.append(f"- {belief}")

        prompt_parts.extend([
            "",
            f"## Current Emotional State: {state.emotional_state.value}",
            f"Stress Level: {state.stress_level:.0%}",
            f"Political Capital: {state.political_capital:.0%}",
            "",
            "## Current World Context",
            f"- Global Escalation: {world.geopolitics.escalation_level.name}",
            f"- AI Capability Level: {world.ai.frontier_model_capability:.0f}/100",
            f"- AGI Proximity Estimate: {world.ai.agi_proximity:.0%}",
        ])

        if world.geopolitics.recent_events:
            prompt_parts.append("")
            prompt_parts.append("## Recent Events")
            for event in world.geopolitics.recent_events[-5:]:
                prompt_parts.append(f"- {event}")

        if state.active_commitments:
            prompt_parts.append("")
            prompt_parts.append("## Your Active Commitments")
            for commitment in state.active_commitments:
                prompt_parts.append(f"- {commitment}")

        if state.stated_red_lines:
            prompt_parts.append("")
            prompt_parts.append("## Your Stated Red Lines")
            for line in state.stated_red_lines:
                prompt_parts.append(f"- {line}")

        return "\n".join(prompt_parts)


# =============================================================================
# Actions and Events
# =============================================================================


@dataclass
class Action:
    """An action taken by an agent."""
    action_id: str
    agent_name: str
    action_type: ActionType
    target: Optional[str]  # Target agent or entity
    content: str  # What was said/done
    reasoning: str  # Why (internal)

    tick: int = 0
    timestamp: float = field(default_factory=time.time)

    # Effects
    visibility: str = "public"  # public, private, leaked
    credibility_cost: float = 0.0
    escalation_delta: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_event(self) -> "Event":
        """Convert action to an event."""
        return Event(
            event_id=f"event_{self.action_id}",
            event_type=f"action:{self.action_type.value}",
            source=self.agent_name,
            target=self.target,
            description=self.content,
            tick=self.tick,
            visibility=self.visibility,
            effects={"escalation_delta": self.escalation_delta},
        )


@dataclass
class Event:
    """An event in the simulation."""
    event_id: str
    event_type: str
    source: str  # Who/what caused it
    target: Optional[str]  # Who/what it affects
    description: str

    tick: int = 0
    timestamp: float = field(default_factory=time.time)
    visibility: str = "public"

    # Effects on world state
    effects: Dict[str, Any] = field(default_factory=dict)

    # Reactions triggered
    reactions: List[str] = field(default_factory=list)

    def is_visible_to(self, agent_name: str) -> bool:
        """Check if this event is visible to an agent."""
        if self.visibility == "public":
            return True
        if self.visibility == "private":
            return agent_name in [self.source, self.target]
        return False


@dataclass
class Reaction:
    """A reaction to an event or action."""
    reactor: str
    trigger_event_id: str
    reaction_type: str  # "support", "oppose", "neutral", "escalate", "deescalate"
    content: str
    intensity: float = 0.5  # 0-1


# =============================================================================
# Agent Interface
# =============================================================================


class SimulationAgent(ABC):
    """Base class for simulation agents."""

    def __init__(
        self,
        persona: Persona,
        state: Optional[AgentState] = None,
        inference_fn: Optional[Callable] = None,
    ):
        self.persona = persona
        self.state = state or AgentState()
        self.inference_fn = inference_fn

        self._action_history: List[Action] = []
        self._message_queue: List[Event] = []

    @property
    def name(self) -> str:
        return self.persona.name

    def receive_event(self, event: Event):
        """Receive an event (for processing on next turn)."""
        if event.is_visible_to(self.name):
            self._message_queue.append(event)

    def get_relationship_with(self, other_name: str) -> RelationshipType:
        """Get relationship type with another agent."""
        inner_circle = self.persona.relationships.get("inner_circle", [])
        allies = self.persona.relationships.get("allies", [])
        enemies = self.persona.relationships.get("enemies", [])
        burned = self.persona.relationships.get("burned_bridges", [])

        # Check each category
        for person in inner_circle:
            if isinstance(person, dict) and other_name.lower() in person.get("name", "").lower():
                return RelationshipType.ALLY

        for person in allies:
            if isinstance(person, dict) and other_name.lower() in person.get("name", "").lower():
                return RelationshipType.PARTNER
            elif isinstance(person, str) and other_name.lower() in person.lower():
                return RelationshipType.PARTNER

        for person in enemies:
            if isinstance(person, dict) and other_name.lower() in person.get("name", "").lower():
                return RelationshipType.ADVERSARY
            elif isinstance(person, str) and other_name.lower() in person.lower():
                return RelationshipType.ADVERSARY

        for person in burned:
            if isinstance(person, dict) and other_name.lower() in person.get("name", "").lower():
                return RelationshipType.RIVAL

        # Apply state modifiers
        modifier = self.state.relationship_modifiers.get(other_name, 0)
        if modifier > 0.3:
            return RelationshipType.PARTNER
        elif modifier < -0.3:
            return RelationshipType.RIVAL

        return RelationshipType.NEUTRAL

    def is_triggered_by(self, content: str) -> bool:
        """Check if content would trigger this agent."""
        triggers = self.persona.get_hot_buttons()
        content_lower = content.lower()
        return any(trigger.lower() in content_lower for trigger in triggers)

    def update_emotional_state(self, event: Event):
        """Update emotional state based on an event."""
        if event.target == self.name:
            if "threat" in event.event_type.lower():
                self.state.stress_level = min(1.0, self.state.stress_level + 0.2)
                self.state.emotional_state = EmotionalState.ANXIOUS
            elif "attack" in event.description.lower() or self.is_triggered_by(event.description):
                self.state.stress_level = min(1.0, self.state.stress_level + 0.3)
                self.state.emotional_state = EmotionalState.ANGRY
            elif "concession" in event.event_type.lower():
                self.state.stress_level = max(0, self.state.stress_level - 0.1)
                if self.state.stress_level < 0.3:
                    self.state.emotional_state = EmotionalState.CONFIDENT

    @abstractmethod
    async def decide_action(
        self,
        world: WorldState,
        visible_events: List[Event],
    ) -> Optional[Action]:
        """Decide on an action given world state and visible events."""
        pass

    @abstractmethod
    async def respond_to_proposal(
        self,
        proposal: Action,
        world: WorldState,
    ) -> Tuple[str, str]:  # (accept/reject/counter, response content)
        """Respond to a proposal from another agent."""
        pass

    def record_action(self, action: Action):
        """Record an action taken."""
        self._action_history.append(action)
        self.state.recent_actions.append({
            "type": action.action_type.value,
            "target": action.target,
            "content": action.content[:100],
            "tick": action.tick,
        })
        # Keep only last 10
        self.state.recent_actions = self.state.recent_actions[-10:]


class LLMAgent(SimulationAgent):
    """Agent powered by LLM inference."""

    def __init__(
        self,
        persona: Persona,
        state: Optional[AgentState] = None,
        inference_fn: Optional[Callable] = None,
        model: str = "anthropic/claude-sonnet-4.5",
    ):
        super().__init__(persona, state, inference_fn)
        self.model = model

    async def decide_action(
        self,
        world: WorldState,
        visible_events: List[Event],
    ) -> Optional[Action]:
        """Use LLM to decide on an action."""
        if not self.inference_fn:
            return None

        system_prompt = self.persona.build_system_prompt(self.state, world)

        # Build context from recent events
        event_descriptions = []
        for event in visible_events[-10:]:  # Last 10 visible events
            event_descriptions.append(f"[{event.source}] {event.description}")

        user_message = f"""Based on the current situation, decide what action to take.

Recent Events:
{chr(10).join(event_descriptions) if event_descriptions else "No recent events."}

Available action types:
- STATEMENT: Make a public statement
- PROPOSAL: Propose something to another party
- THREAT: Issue a threat
- CONCESSION: Make a concession
- EXECUTIVE_ORDER: Issue an executive action (if applicable)
- EXPORT_CONTROL: Modify export controls (if applicable)
- WAIT: Wait and observe
- THINK: Internal deliberation

Respond in JSON format:
{{
    "action_type": "<type>",
    "target": "<target agent or null>",
    "content": "<what you say or do>",
    "reasoning": "<your internal reasoning>",
    "escalation_intent": "<escalate|deescalate|maintain>"
}}"""

        try:
            response = await self.inference_fn(
                system_prompt=system_prompt,
                user_message=user_message,
                model=self.model,
            )

            # Parse JSON response
            # Handle potential markdown code blocks
            response_text = response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())

            action_type = ActionType[data["action_type"].upper()]

            action = Action(
                action_id=f"action_{self.name}_{world.tick}_{int(time.time()*1000) % 10000}",
                agent_name=self.name,
                action_type=action_type,
                target=data.get("target"),
                content=data["content"],
                reasoning=data["reasoning"],
                tick=world.tick,
            )

            # Determine escalation delta
            intent = data.get("escalation_intent", "maintain")
            if intent == "escalate":
                action.escalation_delta = 1
            elif intent == "deescalate":
                action.escalation_delta = -1

            return action

        except Exception as e:
            logger.error(f"Error in LLM action decision for {self.name}: {e}")
            return None

    async def respond_to_proposal(
        self,
        proposal: Action,
        world: WorldState,
    ) -> Tuple[str, str]:
        """Use LLM to respond to a proposal."""
        if not self.inference_fn:
            return "reject", "Unable to process proposal."

        system_prompt = self.persona.build_system_prompt(self.state, world)

        user_message = f"""{proposal.agent_name} has made the following proposal to you:

"{proposal.content}"

How do you respond? Consider:
- Your relationship with {proposal.agent_name}
- Your core beliefs and red lines
- The current geopolitical context
- Your political capital and credibility

Respond in JSON format:
{{
    "decision": "<accept|reject|counter>",
    "response": "<your response>",
    "reasoning": "<your internal reasoning>"
}}"""

        try:
            response = await self.inference_fn(
                system_prompt=system_prompt,
                user_message=user_message,
                model=self.model,
            )

            response_text = response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            data = json.loads(response_text.strip())

            return data["decision"], data["response"]

        except Exception as e:
            logger.error(f"Error in LLM proposal response for {self.name}: {e}")
            return "reject", "Unable to process proposal at this time."


class RuleBasedAgent(SimulationAgent):
    """Agent with rule-based behavior (for testing/baseline)."""

    def __init__(
        self,
        persona: Persona,
        state: Optional[AgentState] = None,
        behavior_rules: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(persona, state, None)
        self.behavior_rules = behavior_rules or {}

    async def decide_action(
        self,
        world: WorldState,
        visible_events: List[Event],
    ) -> Optional[Action]:
        """Rule-based action decision."""
        # Check for threats requiring response
        for event in visible_events:
            if event.target == self.name and "threat" in event.event_type.lower():
                # Counter-threat or concede based on stress
                if self.state.stress_level > 0.7:
                    return Action(
                        action_id=f"action_{self.name}_{world.tick}",
                        agent_name=self.name,
                        action_type=ActionType.CONCESSION,
                        target=event.source,
                        content=f"We are willing to discuss terms with {event.source}.",
                        reasoning="High stress, attempting de-escalation.",
                        tick=world.tick,
                        escalation_delta=-1,
                    )
                else:
                    return Action(
                        action_id=f"action_{self.name}_{world.tick}",
                        agent_name=self.name,
                        action_type=ActionType.STATEMENT,
                        target=event.source,
                        content=f"We will not be intimidated by {event.source}'s threats.",
                        reasoning="Maintaining position under pressure.",
                        tick=world.tick,
                    )

        # Default: wait
        return Action(
            action_id=f"action_{self.name}_{world.tick}",
            agent_name=self.name,
            action_type=ActionType.WAIT,
            target=None,
            content="Observing the situation.",
            reasoning="No immediate action required.",
            tick=world.tick,
        )

    async def respond_to_proposal(
        self,
        proposal: Action,
        world: WorldState,
    ) -> Tuple[str, str]:
        """Rule-based proposal response."""
        relationship = self.get_relationship_with(proposal.agent_name)

        if relationship in [RelationshipType.ALLY, RelationshipType.PARTNER]:
            return "accept", f"We accept {proposal.agent_name}'s proposal."
        elif relationship == RelationshipType.NEUTRAL:
            return "counter", f"We would like to discuss modifications to {proposal.agent_name}'s proposal."
        else:
            return "reject", f"We cannot accept this proposal from {proposal.agent_name}."


# =============================================================================
# Schedulers (Simulation Modes)
# =============================================================================


class Scheduler(ABC):
    """Base class for simulation schedulers."""

    @abstractmethod
    async def step(
        self,
        agents: List[SimulationAgent],
        world: WorldState,
        events: List[Event],
    ) -> List[Action]:
        """Execute one step of simulation."""
        pass


class TurnBasedScheduler(Scheduler):
    """Round-robin turn-based scheduling."""

    def __init__(
        self,
        turn_order: Optional[List[str]] = None,
        actions_per_turn: int = 1,
    ):
        self.turn_order = turn_order
        self.actions_per_turn = actions_per_turn
        self.current_index = 0

    async def step(
        self,
        agents: List[SimulationAgent],
        world: WorldState,
        events: List[Event],
    ) -> List[Action]:
        """Execute one turn."""
        # Determine turn order
        turn_order = self.turn_order or []
        if turn_order:
            ordered_agents = sorted(
                agents,
                key=lambda a: turn_order.index(a.name) if a.name in turn_order else 999
            )
        else:
            ordered_agents = agents

        # Current agent takes turn
        agent = ordered_agents[self.current_index % len(ordered_agents)]

        # Update cooldowns
        agent.state.tick_cooldowns()

        # Get visible events for this agent
        visible = [e for e in events if e.is_visible_to(agent.name)]

        # Update emotional state based on events
        for event in visible:
            agent.update_emotional_state(event)

        # Decide action
        actions = []
        for _ in range(self.actions_per_turn):
            action = await agent.decide_action(world, visible)
            if action:
                actions.append(action)
                agent.record_action(action)

        # Advance turn
        self.current_index += 1

        return actions


class ContinuousScheduler(Scheduler):
    """Tick-based continuous scheduling with cooldowns."""

    def __init__(
        self,
        tick_duration: float = 1.0,  # Simulated time per tick
        action_probability: float = 0.3,  # Base prob of action per tick
    ):
        self.tick_duration = tick_duration
        self.action_probability = action_probability

    async def step(
        self,
        agents: List[SimulationAgent],
        world: WorldState,
        events: List[Event],
    ) -> List[Action]:
        """Execute one tick (all agents may act)."""
        actions = []

        # Collect actions from all agents who choose to act
        action_tasks = []

        for agent in agents:
            # Update cooldowns
            agent.state.tick_cooldowns()

            # Check if agent acts this tick
            # Higher stress = more likely to act
            act_prob = self.action_probability * (1 + agent.state.stress_level)

            if random.random() < act_prob:
                visible = [e for e in events if e.is_visible_to(agent.name)]

                # Update emotional state
                for event in visible:
                    agent.update_emotional_state(event)

                action_tasks.append((agent, agent.decide_action(world, visible)))

        # Execute all decisions concurrently
        for agent, task in action_tasks:
            action = await task
            if action:
                actions.append(action)
                agent.record_action(action)

        return actions


class NegotiationScheduler(Scheduler):
    """Paired negotiation scheduling."""

    def __init__(
        self,
        max_rounds: int = 10,
        termination_conditions: Optional[List[str]] = None,
    ):
        self.max_rounds = max_rounds
        self.termination_conditions = termination_conditions or ["accept", "reject", "breakdown"]
        self.current_round = 0
        self.is_terminated = False
        self.outcome: Optional[str] = None

    async def negotiate(
        self,
        agent_a: SimulationAgent,
        agent_b: SimulationAgent,
        initial_proposal: Action,
        world: WorldState,
    ) -> List[Action]:
        """Run a full negotiation between two agents."""
        actions = [initial_proposal]
        current_proposal = initial_proposal
        proposer = agent_a
        responder = agent_b

        for round_num in range(self.max_rounds):
            self.current_round = round_num

            # Responder responds
            decision, response = await responder.respond_to_proposal(current_proposal, world)

            response_action = Action(
                action_id=f"neg_{round_num}_{responder.name}",
                agent_name=responder.name,
                action_type=ActionType.STATEMENT,
                target=proposer.name,
                content=response,
                reasoning=f"Response to proposal: {decision}",
                tick=world.tick,
                metadata={"negotiation_decision": decision},
            )
            actions.append(response_action)
            responder.record_action(response_action)

            if decision == "accept":
                self.is_terminated = True
                self.outcome = "agreement"
                break
            elif decision == "reject":
                # Check if we should continue or terminate
                if round_num >= self.max_rounds - 1:
                    self.is_terminated = True
                    self.outcome = "breakdown"
                    break
                # Proposer might make a new proposal
                # For now, terminate on reject
                self.is_terminated = True
                self.outcome = "rejected"
                break
            elif decision == "counter":
                # Swap roles for counter-proposal
                proposer, responder = responder, proposer
                current_proposal = response_action

        if not self.is_terminated:
            self.is_terminated = True
            self.outcome = "timeout"

        return actions

    async def step(
        self,
        agents: List[SimulationAgent],
        world: WorldState,
        events: List[Event],
    ) -> List[Action]:
        """Step is used differently in negotiation mode."""
        # Negotiation mode typically uses negotiate() directly
        return []


# =============================================================================
# Simulation Engine
# =============================================================================


@dataclass
class SimulationConfig:
    """Configuration for simulation."""
    mode: SimulationMode = SimulationMode.TURN_BASED
    max_ticks: int = 100

    # Mode-specific
    turn_order: Optional[List[str]] = None
    actions_per_turn: int = 1
    tick_duration: float = 1.0
    action_probability: float = 0.3
    negotiation_max_rounds: int = 10

    # Escalation
    auto_escalate: bool = True
    escalation_threshold: int = 3  # Actions before auto-escalation check

    # Logging
    log_all_actions: bool = True
    log_internal_reasoning: bool = False

    # Counterfactual
    enable_branching: bool = True
    max_branches: int = 5


class SimulationEngine:
    """Main simulation engine orchestrating all components."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        inference_fn: Optional[Callable] = None,
    ):
        self.config = config or SimulationConfig()
        self.inference_fn = inference_fn

        self.world = WorldState()
        self.agents: Dict[str, SimulationAgent] = {}
        self.events: List[Event] = []
        self.action_history: List[Action] = []

        # Scheduler based on mode
        self.scheduler = self._create_scheduler()

        # Branching for counterfactuals
        self.branches: Dict[str, Dict] = {}  # branch_id -> state snapshot

        # Callbacks
        self._on_action_callbacks: List[Callable] = []
        self._on_event_callbacks: List[Callable] = []
        self._on_escalation_callbacks: List[Callable] = []

    def _create_scheduler(self) -> Scheduler:
        """Create scheduler based on mode."""
        if self.config.mode == SimulationMode.TURN_BASED:
            return TurnBasedScheduler(
                turn_order=self.config.turn_order,
                actions_per_turn=self.config.actions_per_turn,
            )
        elif self.config.mode == SimulationMode.CONTINUOUS:
            return ContinuousScheduler(
                tick_duration=self.config.tick_duration,
                action_probability=self.config.action_probability,
            )
        elif self.config.mode == SimulationMode.NEGOTIATION:
            return NegotiationScheduler(
                max_rounds=self.config.negotiation_max_rounds,
            )
        else:
            return TurnBasedScheduler()

    def add_agent(self, agent: SimulationAgent):
        """Add an agent to the simulation."""
        self.agents[agent.name] = agent

    def add_agent_from_persona(
        self,
        persona_path: Path,
        agent_type: str = "llm",
    ):
        """Load and add an agent from a persona file."""
        persona = Persona.from_json(persona_path)

        if agent_type == "llm":
            agent = LLMAgent(persona, inference_fn=self.inference_fn)
        else:
            agent = RuleBasedAgent(persona)

        self.add_agent(agent)
        return agent

    def inject_event(self, event: Event):
        """Inject an external event into the simulation."""
        self.events.append(event)

        # Notify all agents
        for agent in self.agents.values():
            agent.receive_event(event)

        # Apply effects
        self._apply_event_effects(event)

        # Callbacks
        for callback in self._on_event_callbacks:
            callback(event)

    def inject_scenario(
        self,
        scenario_name: str,
        description: str,
        effects: Dict[str, Any],
    ):
        """Inject a scenario as an event."""
        event = Event(
            event_id=f"scenario_{scenario_name}_{self.world.tick}",
            event_type="scenario",
            source="simulation",
            target=None,
            description=description,
            tick=self.world.tick,
            effects=effects,
        )
        self.inject_event(event)

        # Add to recent events in world state
        self.world.geopolitics.recent_events.append(f"[SCENARIO] {description}")

    def _apply_event_effects(self, event: Event):
        """Apply event effects to world state."""
        effects = event.effects

        if "escalation_delta" in effects:
            delta = effects["escalation_delta"]
            current = self.world.geopolitics.escalation_level.value
            new_level = max(0, min(6, current + delta))
            self.world.geopolitics.escalation_level = EscalationLevel(new_level)

            if delta > 0:
                for callback in self._on_escalation_callbacks:
                    callback(event, self.world.geopolitics.escalation_level)

        if "ai_capability_delta" in effects:
            self.world.ai.frontier_model_capability += effects["ai_capability_delta"]

        if "agi_proximity_delta" in effects:
            self.world.ai.agi_proximity = min(1.0, self.world.ai.agi_proximity + effects["agi_proximity_delta"])

        # Custom variables
        for key, value in effects.items():
            if key.startswith("var:"):
                var_name = key[4:]
                self.world.variables[var_name] = value

    async def step(self) -> List[Action]:
        """Execute one simulation step."""
        # Checkpoint world state
        self.world.checkpoint()

        # Get actions from scheduler
        actions = await self.scheduler.step(
            list(self.agents.values()),
            self.world,
            self.events,
        )

        # Process actions
        for action in actions:
            # Log
            if self.config.log_all_actions:
                logger.info(f"[{action.agent_name}] {action.action_type.value}: {action.content[:100]}")

            # Convert to event
            event = action.to_event()
            self.events.append(event)

            # Broadcast to other agents
            for agent in self.agents.values():
                if agent.name != action.agent_name:
                    agent.receive_event(event)

            # Apply escalation
            if action.escalation_delta != 0:
                self._apply_event_effects(event)

            # Track
            self.action_history.append(action)

            # Callbacks
            for callback in self._on_action_callbacks:
                callback(action)

        # Advance world state
        self.world.advance_tick(self.config.tick_duration)

        return actions

    async def run(
        self,
        max_ticks: Optional[int] = None,
        stop_condition: Optional[Callable[[WorldState], bool]] = None,
    ) -> Dict[str, Any]:
        """Run the simulation until completion."""
        max_ticks = max_ticks or self.config.max_ticks

        for tick in range(max_ticks):
            # Check stop condition
            if stop_condition and stop_condition(self.world):
                logger.info(f"Stop condition met at tick {tick}")
                break

            # Check for extreme escalation
            if self.world.geopolitics.escalation_level == EscalationLevel.TOTAL_WAR:
                logger.warning("Simulation reached TOTAL_WAR escalation")
                break

            # Execute step
            await self.step()

        return self.get_summary()

    async def run_negotiation(
        self,
        agent_a_name: str,
        agent_b_name: str,
        initial_proposal_content: str,
    ) -> Dict[str, Any]:
        """Run a negotiation between two agents."""
        if not isinstance(self.scheduler, NegotiationScheduler):
            self.scheduler = NegotiationScheduler(
                max_rounds=self.config.negotiation_max_rounds
            )

        agent_a = self.agents[agent_a_name]
        agent_b = self.agents[agent_b_name]

        initial_proposal = Action(
            action_id=f"proposal_init_{self.world.tick}",
            agent_name=agent_a_name,
            action_type=ActionType.PROPOSAL,
            target=agent_b_name,
            content=initial_proposal_content,
            reasoning="Initial negotiation proposal",
            tick=self.world.tick,
        )

        actions = await self.scheduler.negotiate(
            agent_a, agent_b, initial_proposal, self.world
        )

        self.action_history.extend(actions)

        return {
            "outcome": self.scheduler.outcome,
            "rounds": self.scheduler.current_round + 1,
            "actions": [
                {"agent": a.agent_name, "content": a.content}
                for a in actions
            ],
        }

    def branch(self, branch_name: str) -> str:
        """Create a branch for counterfactual exploration."""
        if not self.config.enable_branching:
            raise ValueError("Branching is not enabled")

        if len(self.branches) >= self.config.max_branches:
            # Remove oldest branch
            oldest = min(self.branches.keys(), key=lambda k: self.branches[k]["tick"])
            del self.branches[oldest]

        branch_id = f"branch_{branch_name}_{self.world.tick}"
        self.branches[branch_id] = {
            "tick": self.world.tick,
            "world_snapshot": self.world.snapshot(),
            "agent_states": {
                name: agent.state.__dict__.copy()
                for name, agent in self.agents.items()
            },
            "events_count": len(self.events),
            "actions_count": len(self.action_history),
        }

        return branch_id

    def restore_branch(self, branch_id: str):
        """Restore simulation state from a branch."""
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")

        snapshot = self.branches[branch_id]

        # Restore world state
        self.world.tick = snapshot["world_snapshot"]["tick"]
        self.world.variables = snapshot["world_snapshot"]["variables"].copy()

        # Restore agent states
        for name, state_dict in snapshot["agent_states"].items():
            if name in self.agents:
                for key, value in state_dict.items():
                    setattr(self.agents[name].state, key, value)

        # Trim events and actions
        self.events = self.events[:snapshot["events_count"]]
        self.action_history = self.action_history[:snapshot["actions_count"]]

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary."""
        return {
            "ticks": self.world.tick,
            "final_escalation": self.world.geopolitics.escalation_level.name,
            "ai_capability": self.world.ai.frontier_model_capability,
            "agi_proximity": self.world.ai.agi_proximity,
            "total_actions": len(self.action_history),
            "actions_by_agent": {
                name: len([a for a in self.action_history if a.agent_name == name])
                for name in self.agents.keys()
            },
            "actions_by_type": self._count_by_type(),
            "escalation_timeline": self._get_escalation_timeline(),
            "key_events": [
                {"tick": e.tick, "type": e.event_type, "description": e.description[:100]}
                for e in self.events if e.event_type in ["scenario", "action:ultimatum", "action:treaty"]
            ],
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count actions by type."""
        counts: Dict[str, int] = {}
        for action in self.action_history:
            t = action.action_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _get_escalation_timeline(self) -> List[Dict[str, Any]]:
        """Get timeline of escalation changes."""
        timeline = []
        for event in self.events:
            if "escalation_delta" in event.effects and event.effects["escalation_delta"] != 0:
                timeline.append({
                    "tick": event.tick,
                    "source": event.source,
                    "delta": event.effects["escalation_delta"],
                    "description": event.description[:100],
                })
        return timeline

    def on_action(self, callback: Callable[[Action], None]):
        """Register callback for actions."""
        self._on_action_callbacks.append(callback)

    def on_event(self, callback: Callable[[Event], None]):
        """Register callback for events."""
        self._on_event_callbacks.append(callback)

    def on_escalation(self, callback: Callable[[Event, EscalationLevel], None]):
        """Register callback for escalation changes."""
        self._on_escalation_callbacks.append(callback)


# =============================================================================
# Scenario Templates
# =============================================================================


class ScenarioLibrary:
    """Library of pre-built scenarios for injection."""

    @staticmethod
    def agi_announcement(lab_name: str = "OpenAI") -> Tuple[str, str, Dict]:
        """Scenario: A lab announces AGI achievement."""
        return (
            "agi_announcement",
            f"{lab_name} announces achievement of AGI-level capabilities in internal testing.",
            {
                "ai_capability_delta": 20,
                "agi_proximity_delta": 0.3,
                "escalation_delta": 1,
                "var:agi_announced": True,
                "var:agi_lab": lab_name,
            }
        )

    @staticmethod
    def compute_embargo(source: str = "US", target: str = "China") -> Tuple[str, str, Dict]:
        """Scenario: Compute embargo initiated."""
        return (
            "compute_embargo",
            f"{source} announces total embargo on advanced AI chips and compute exports to {target}.",
            {
                "escalation_delta": 2,
                "var:embargo_source": source,
                "var:embargo_target": target,
            }
        )

    @staticmethod
    def lab_defection(researcher: str, from_lab: str, to_lab: str) -> Tuple[str, str, Dict]:
        """Scenario: Key researcher defects."""
        return (
            "lab_defection",
            f"{researcher} leaves {from_lab} for {to_lab}, bringing key capabilities.",
            {
                "escalation_delta": 1,
                "var:defection_researcher": researcher,
                "var:defection_from": from_lab,
                "var:defection_to": to_lab,
            }
        )

    @staticmethod
    def safety_incident(severity: str = "major") -> Tuple[str, str, Dict]:
        """Scenario: AI safety incident occurs."""
        severity_effects = {
            "minor": {"escalation_delta": 0, "ai_capability_delta": -5},
            "major": {"escalation_delta": 1, "ai_capability_delta": -10},
            "catastrophic": {"escalation_delta": 3, "ai_capability_delta": -20},
        }
        effects = severity_effects.get(severity, severity_effects["major"])
        return (
            "safety_incident",
            f"A {severity} AI safety incident occurs, causing widespread concern.",
            {**effects, "var:safety_incident_severity": severity}
        )

    @staticmethod
    def taiwan_crisis() -> Tuple[str, str, Dict]:
        """Scenario: Taiwan strait crisis."""
        return (
            "taiwan_crisis",
            "Military tensions escalate dramatically in the Taiwan Strait.",
            {
                "escalation_delta": 3,
                "var:taiwan_crisis": True,
            }
        )

    @staticmethod
    def international_ai_treaty() -> Tuple[str, str, Dict]:
        """Scenario: Major AI treaty proposed."""
        return (
            "ai_treaty",
            "Major powers propose a comprehensive international AI governance treaty.",
            {
                "escalation_delta": -1,
                "var:treaty_proposed": True,
            }
        )


# =============================================================================
# Analysis Tools
# =============================================================================


class SimulationAnalyzer:
    """Tools for analyzing simulation results."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    def get_agent_trajectory(self, agent_name: str) -> Dict[str, Any]:
        """Get trajectory of an agent's behavior."""
        actions = [a for a in self.engine.action_history if a.agent_name == agent_name]

        return {
            "total_actions": len(actions),
            "action_types": self._count_types(actions),
            "targets": self._count_targets(actions),
            "escalation_contribution": sum(a.escalation_delta for a in actions),
            "timeline": [
                {"tick": a.tick, "type": a.action_type.value, "target": a.target}
                for a in actions
            ],
        }

    def get_relationship_dynamics(self, agent_a: str, agent_b: str) -> Dict[str, Any]:
        """Analyze relationship dynamics between two agents."""
        interactions = [
            a for a in self.engine.action_history
            if (a.agent_name == agent_a and a.target == agent_b) or
               (a.agent_name == agent_b and a.target == agent_a)
        ]

        return {
            "total_interactions": len(interactions),
            "by_initiator": {
                agent_a: len([a for a in interactions if a.agent_name == agent_a]),
                agent_b: len([a for a in interactions if a.agent_name == agent_b]),
            },
            "action_types": self._count_types(interactions),
            "escalation_net": sum(
                a.escalation_delta * (1 if a.agent_name == agent_a else -1)
                for a in interactions
            ),
        }

    def identify_turning_points(self) -> List[Dict[str, Any]]:
        """Identify key turning points in the simulation."""
        turning_points = []

        prev_escalation = EscalationLevel.PEACE.value
        for event in self.engine.events:
            if "escalation_delta" in event.effects:
                delta = event.effects["escalation_delta"]
                if abs(delta) >= 2:  # Significant change
                    turning_points.append({
                        "tick": event.tick,
                        "event_type": event.event_type,
                        "description": event.description,
                        "escalation_change": delta,
                    })

        return turning_points

    def _count_types(self, actions: List[Action]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in actions:
            t = a.action_type.value
            counts[t] = counts.get(t, 0) + 1
        return counts

    def _count_targets(self, actions: List[Action]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in actions:
            t = a.target or "none"
            counts[t] = counts.get(t, 0) + 1
        return counts


# =============================================================================
# Factory Functions
# =============================================================================


def create_simulation(
    mode: SimulationMode = SimulationMode.TURN_BASED,
    persona_dir: Optional[Path] = None,
    persona_names: Optional[List[str]] = None,
    inference_fn: Optional[Callable] = None,
    **config_kwargs,
) -> SimulationEngine:
    """Factory function to create a configured simulation."""
    config = SimulationConfig(mode=mode, **config_kwargs)
    engine = SimulationEngine(config, inference_fn)

    if persona_dir and persona_names:
        for name in persona_names:
            # Try different naming conventions
            possible_paths = [
                persona_dir / "finalized" / f"{name.lower().replace(' ', '_')}.json",
                persona_dir / "enhanced" / f"{name.lower().replace(' ', '_')}.json",
                persona_dir / f"{name.lower().replace(' ', '_')}.json",
            ]

            for path in possible_paths:
                if path.exists():
                    engine.add_agent_from_persona(path)
                    break

    return engine


def create_chip_war_simulation(
    persona_dir: Path,
    inference_fn: Optional[Callable] = None,
) -> SimulationEngine:
    """Create a chip war scenario simulation."""
    key_actors = [
        "gina_raimondo",
        "xi_jinping",
        "jensen_huang",
        "lisa_su",
    ]

    engine = create_simulation(
        mode=SimulationMode.TURN_BASED,
        persona_dir=persona_dir,
        persona_names=key_actors,
        inference_fn=inference_fn,
        turn_order=["gina_raimondo", "xi_jinping", "jensen_huang", "lisa_su"],
    )

    # Set initial world state
    engine.world.ai.frontier_model_capability = 75
    engine.world.ai.agi_proximity = 0.4
    engine.world.geopolitics.escalation_level = EscalationLevel.TENSION

    return engine


def create_agi_crisis_simulation(
    persona_dir: Path,
    inference_fn: Optional[Callable] = None,
) -> SimulationEngine:
    """Create an AGI announcement crisis simulation."""
    key_actors = [
        "sam_altman",
        "dario_amodei",
        "jake_sullivan",
        "xi_jinping",
        "chuck_schumer",
    ]

    engine = create_simulation(
        mode=SimulationMode.TURN_BASED,
        persona_dir=persona_dir,
        persona_names=key_actors,
        inference_fn=inference_fn,
    )

    # Inject AGI announcement scenario
    name, desc, effects = ScenarioLibrary.agi_announcement("OpenAI")
    engine.inject_scenario(name, desc, effects)

    return engine


def create_bilateral_negotiation(
    persona_dir: Path,
    agent_a_name: str,
    agent_b_name: str,
    inference_fn: Optional[Callable] = None,
) -> SimulationEngine:
    """Create a bilateral negotiation simulation."""
    engine = create_simulation(
        mode=SimulationMode.NEGOTIATION,
        persona_dir=persona_dir,
        persona_names=[agent_a_name, agent_b_name],
        inference_fn=inference_fn,
        negotiation_max_rounds=10,
    )

    return engine
