"""
Advanced Programmatic Scenarios

Defines complex multi-agent scenarios with:
- Phases and state machines
- Triggers and conditions
- Dynamic branching
- Event-driven reactions
- Metrics and outcome tracking
- Scenario composition
"""

from __future__ import annotations

import asyncio
import logging
import random
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Optional, List, Dict, Any, Callable, Awaitable,
    Union, TypeVar, Generic, Set, Tuple
)

from .driver import SimulationDriver, Character, CharacterAction, ActionType, ActionResult

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# Core Types
# =============================================================================

class ScenarioState(str, Enum):
    """State of a scenario execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class PhaseState(str, Enum):
    """State of a scenario phase."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class TriggerType(str, Enum):
    """Types of scenario triggers."""
    ON_START = "on_start"
    ON_PHASE_ENTER = "on_phase_enter"
    ON_PHASE_EXIT = "on_phase_exit"
    ON_ACTION = "on_action"
    ON_MESSAGE = "on_message"
    ON_BELIEF_CHANGE = "on_belief_change"
    ON_CONDITION = "on_condition"
    ON_TIMEOUT = "on_timeout"
    ON_EXTERNAL = "on_external"


class ConditionOperator(str, Enum):
    """Operators for conditions."""
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"
    ALL = "all"
    ANY = "any"
    NONE = "none"


# =============================================================================
# Conditions
# =============================================================================

@dataclass
class Condition:
    """A condition that can be evaluated."""
    field: str  # Path to the field (e.g., "character.beliefs.ai_safety")
    operator: ConditionOperator
    value: Any
    negate: bool = False

    def evaluate(self, context: "ScenarioContext") -> bool:
        """Evaluate the condition against the context."""
        actual = self._get_field_value(context)
        result = self._compare(actual)
        return not result if self.negate else result

    def _get_field_value(self, context: "ScenarioContext") -> Any:
        """Extract field value from context."""
        parts = self.field.split(".")
        current = context

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            elif hasattr(current, '__getitem__'):
                try:
                    current = current[part]
                except (KeyError, IndexError):
                    return None
            else:
                return None

        return current

    def _compare(self, actual: Any) -> bool:
        """Compare actual value with expected."""
        if self.operator == ConditionOperator.EQ:
            return actual == self.value
        elif self.operator == ConditionOperator.NE:
            return actual != self.value
        elif self.operator == ConditionOperator.GT:
            return actual is not None and actual > self.value
        elif self.operator == ConditionOperator.GE:
            return actual is not None and actual >= self.value
        elif self.operator == ConditionOperator.LT:
            return actual is not None and actual < self.value
        elif self.operator == ConditionOperator.LE:
            return actual is not None and actual <= self.value
        elif self.operator == ConditionOperator.IN:
            return actual in self.value
        elif self.operator == ConditionOperator.NOT_IN:
            return actual not in self.value
        elif self.operator == ConditionOperator.CONTAINS:
            return self.value in actual if actual else False
        elif self.operator == ConditionOperator.MATCHES:
            import re
            return bool(re.match(self.value, str(actual))) if actual else False
        return False


@dataclass
class ConditionGroup:
    """Group of conditions with AND/OR logic."""
    conditions: List[Union[Condition, "ConditionGroup"]]
    mode: str = "all"  # "all" (AND) or "any" (OR)

    def evaluate(self, context: "ScenarioContext") -> bool:
        """Evaluate the condition group."""
        results = []
        for c in self.conditions:
            if isinstance(c, dict):
                # Convert dict to Condition
                c = Condition(**c)
            results.append(c.evaluate(context))
        if self.mode == "all":
            return all(results)
        elif self.mode == "any":
            return any(results)
        elif self.mode == "none":
            return not any(results)
        return False

    @classmethod
    def all_of(cls, *conditions) -> "ConditionGroup":
        return cls(list(conditions), mode="all")

    @classmethod
    def any_of(cls, *conditions) -> "ConditionGroup":
        return cls(list(conditions), mode="any")

    @classmethod
    def none_of(cls, *conditions) -> "ConditionGroup":
        return cls(list(conditions), mode="none")


# =============================================================================
# Actions and Effects
# =============================================================================

@dataclass
class ScenarioEffect:
    """An effect that modifies scenario state."""
    effect_type: str
    target: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    async def apply(self, context: "ScenarioContext"):
        """Apply the effect."""
        if self.effect_type == "set_variable":
            context.variables[self.params["name"]] = self.params["value"]
        elif self.effect_type == "increment_variable":
            name = self.params["name"]
            context.variables[name] = context.variables.get(name, 0) + self.params.get("amount", 1)
        elif self.effect_type == "update_belief":
            char = context.get_character(self.target)
            if char:
                char.beliefs[self.params["topic"]] = self.params["value"]
        elif self.effect_type == "add_to_coalition":
            coalition = self.params.get("coalition", "default")
            if coalition not in context.coalitions:
                context.coalitions[coalition] = set()
            context.coalitions[coalition].add(self.target)
        elif self.effect_type == "remove_from_coalition":
            coalition = self.params.get("coalition", "default")
            if coalition in context.coalitions:
                context.coalitions[coalition].discard(self.target)
        elif self.effect_type == "emit_event":
            context.emit_event(self.params.get("event_type"), self.params.get("data", {}))
        elif self.effect_type == "log":
            context.log(self.params.get("message", ""))


@dataclass
class ScenarioAction:
    """An action to be executed in a scenario."""
    action_type: str
    character_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    conditions: Optional[ConditionGroup] = None
    effects: List[ScenarioEffect] = field(default_factory=list)
    delay: float = 0.0

    async def execute(self, context: "ScenarioContext") -> Optional[ActionResult]:
        """Execute the action."""
        # Check conditions
        if self.conditions and not self.conditions.evaluate(context):
            return None

        # Apply delay
        if self.delay > 0:
            await asyncio.sleep(self.delay)

        result = None

        # Execute based on type
        if self.action_type == "character_action":
            char_id = self._resolve_character(context)
            if char_id:
                action = CharacterAction(
                    action_type=ActionType(self.params.get("type", "say")),
                    content=self._resolve_template(self.params.get("content", ""), context),
                    target=self._resolve_character(context, self.params.get("target")),
                    context=self.params.get("context", {}),
                )
                result = await context.driver.do(char_id, action)

        elif self.action_type == "broadcast":
            message = self._resolve_template(self.params.get("message", ""), context)
            for char_id in context.participants:
                await context.driver.say(char_id, message, self.params.get("topic", "general"))

        elif self.action_type == "dialogue_round":
            for char_id in self._resolve_character_list(context):
                topic = self.params.get("topic", "discussion")
                await context.driver.think(char_id, topic)
                await context.driver.say(char_id, f"My thoughts on {topic}", topic)

        elif self.action_type == "vote":
            result = await self._execute_vote(context)

        elif self.action_type == "negotiate":
            result = await self._execute_negotiation(context)

        elif self.action_type == "form_coalition":
            result = await self._execute_coalition_formation(context)

        elif self.action_type == "apply_pressure":
            result = await self._execute_pressure(context)

        elif self.action_type == "reveal_information":
            result = await self._execute_reveal(context)

        elif self.action_type == "wait":
            await asyncio.sleep(self.params.get("duration", 1.0))

        elif self.action_type == "parallel":
            # Execute multiple actions in parallel
            sub_actions = self.params.get("actions", [])
            tasks = [ScenarioAction(**a).execute(context) for a in sub_actions]
            await asyncio.gather(*tasks)

        elif self.action_type == "sequence":
            # Execute multiple actions in sequence
            sub_actions = self.params.get("actions", [])
            for a in sub_actions:
                await ScenarioAction(**a).execute(context)

        elif self.action_type == "conditional":
            # Execute based on condition
            condition = ConditionGroup(**self.params.get("condition", {}))
            if condition.evaluate(context):
                then_action = self.params.get("then")
                if then_action:
                    await ScenarioAction(**then_action).execute(context)
            else:
                else_action = self.params.get("else")
                if else_action:
                    await ScenarioAction(**else_action).execute(context)

        elif self.action_type == "loop":
            # Execute action multiple times
            count = self.params.get("count", 1)
            action_def = self.params.get("action")
            for i in range(count):
                context.variables["loop_index"] = i
                if action_def:
                    await ScenarioAction(**action_def).execute(context)

        elif self.action_type == "foreach":
            # Execute for each character in a group
            char_list = self._resolve_character_list(context)
            action_def = self.params.get("action")
            for char_id in char_list:
                context.variables["current_character"] = char_id
                if action_def:
                    await ScenarioAction(**action_def).execute(context)

        # Apply effects
        for effect in self.effects:
            await effect.apply(context)

        return result

    def _resolve_character(self, context: "ScenarioContext", char_ref: str = None) -> Optional[str]:
        """Resolve a character reference to an ID."""
        ref = char_ref or self.character_id
        if not ref:
            return None

        if ref.startswith("$"):
            # Variable reference
            return context.variables.get(ref[1:])
        elif ref.startswith("@"):
            # Role reference
            role = ref[1:]
            return context.roles.get(role)
        elif ref == "random":
            return random.choice(list(context.participants)) if context.participants else None
        elif ref == "current":
            return context.variables.get("current_character")
        return ref

    def _resolve_character_list(self, context: "ScenarioContext") -> List[str]:
        """Resolve a list of characters."""
        chars = self.params.get("characters", [])
        if chars == "all":
            return list(context.participants)
        elif chars == "coalition":
            coalition = self.params.get("coalition", "default")
            return list(context.coalitions.get(coalition, set()))
        elif isinstance(chars, str) and chars.startswith("@"):
            role = chars[1:]
            return [context.roles.get(role)] if role in context.roles else []
        return [self._resolve_character(context, c) for c in chars]

    def _resolve_template(self, template: Any, context: "ScenarioContext") -> Any:
        """Resolve template variables in a string or return non-strings as-is."""
        if not isinstance(template, str):
            return template
        result = template
        for key, value in context.variables.items():
            if isinstance(value, str):
                result = result.replace(f"${{{key}}}", value)
            elif not isinstance(value, (dict, list)):
                result = result.replace(f"${{{key}}}", str(value))
        return result

    async def _execute_vote(self, context: "ScenarioContext") -> Dict[str, Any]:
        """Execute a voting action."""
        proposal = self.params.get("proposal", "")
        voters = self._resolve_character_list(context)

        votes = {}
        for voter_id in voters:
            char = context.get_character(voter_id)
            if not char:
                continue

            # Decision based on beliefs
            topic = self.params.get("topic", "general")
            belief = char.beliefs.get(topic, 0.5)
            threshold = self.params.get("threshold", 0.5)
            noise = random.gauss(0, 0.1)

            decision = "approve" if (belief + noise) > threshold else "reject"
            confidence = abs(belief - threshold)

            votes[voter_id] = {
                "decision": decision,
                "confidence": confidence,
                "belief": belief,
            }

            await context.driver.do(voter_id, CharacterAction.vote(
                proposal_id=context.variables.get("proposal_id", "unknown"),
                decision=decision,
                rationale=f"Based on my {topic} belief of {belief:.2f}"
            ))

        # Tally results
        approve = sum(1 for v in votes.values() if v["decision"] == "approve")
        reject = len(votes) - approve
        passed = approve > reject

        context.variables["last_vote"] = {
            "votes": votes,
            "approve": approve,
            "reject": reject,
            "passed": passed,
        }

        return context.variables["last_vote"]

    async def _execute_negotiation(self, context: "ScenarioContext") -> Dict[str, Any]:
        """Execute a negotiation action."""
        parties = self._resolve_character_list(context)
        rounds = self.params.get("rounds", 3)
        topic = self.params.get("topic", "negotiation")

        offers = []
        for round_num in range(rounds):
            for party_id in parties:
                char = context.get_character(party_id)
                if not char:
                    continue

                # Make offer based on beliefs and previous offers
                base_position = char.beliefs.get(topic, 0.5)
                concession = round_num * 0.1 * random.random()  # Increasing concessions
                offer = base_position - concession if base_position > 0.5 else base_position + concession

                offers.append({
                    "round": round_num,
                    "party": party_id,
                    "offer": offer,
                })

                await context.driver.say(
                    party_id,
                    f"[Round {round_num + 1}] My position: {offer:.2f}",
                    f"negotiation:{topic}"
                )

        # Check for convergence
        if len(offers) >= 2:
            final_offers = [o["offer"] for o in offers[-len(parties):]]
            spread = max(final_offers) - min(final_offers)
            agreement_reached = spread < self.params.get("agreement_threshold", 0.2)
        else:
            agreement_reached = False

        result = {
            "offers": offers,
            "agreement_reached": agreement_reached,
            "final_position": sum(o["offer"] for o in offers[-len(parties):]) / len(parties) if offers else 0,
        }

        context.variables["last_negotiation"] = result
        return result

    async def _execute_coalition_formation(self, context: "ScenarioContext") -> Dict[str, Any]:
        """Execute coalition formation."""
        candidates = self._resolve_character_list(context)
        topic = self.params.get("topic", "coalition")
        coalition_name = self.params.get("coalition_name", f"coalition_{uuid.uuid4().hex[:6]}")

        # Characters with similar beliefs form coalitions
        coalition = set()
        threshold = self.params.get("similarity_threshold", 0.3)

        for i, char_id in enumerate(candidates):
            char = context.get_character(char_id)
            if not char:
                continue

            if not coalition:
                coalition.add(char_id)
                continue

            # Check similarity with existing coalition members
            char_belief = char.beliefs.get(topic, 0.5)
            similarities = []
            for member_id in coalition:
                member = context.get_character(member_id)
                if member:
                    member_belief = member.beliefs.get(topic, 0.5)
                    similarities.append(abs(char_belief - member_belief))

            avg_similarity = sum(similarities) / len(similarities) if similarities else 1.0
            if avg_similarity < threshold:
                coalition.add(char_id)

        context.coalitions[coalition_name] = coalition

        result = {
            "coalition_name": coalition_name,
            "members": list(coalition),
            "size": len(coalition),
        }

        context.variables["last_coalition"] = result
        return result

    async def _execute_pressure(self, context: "ScenarioContext") -> Dict[str, Any]:
        """Execute pressure application on a character."""
        target_id = self._resolve_character(context, self.params.get("target"))
        source_id = self._resolve_character(context, self.params.get("source"))
        topic = self.params.get("topic", "general")
        intensity = self.params.get("intensity", 0.3)

        target = context.get_character(target_id)
        if not target:
            return {"success": False, "error": "Target not found"}

        # Pressure affects beliefs
        current_belief = target.beliefs.get(topic, 0.5)
        direction = self.params.get("direction", 1)  # 1 = increase, -1 = decrease
        resistance = random.random() * 0.5  # Random resistance factor

        new_belief = current_belief + (direction * intensity * (1 - resistance))
        new_belief = max(0.0, min(1.0, new_belief))

        target.beliefs[topic] = new_belief

        # Source applies pressure
        if source_id:
            await context.driver.tell(
                source_id, target_id,
                f"I urge you to reconsider your position on {topic}"
            )

        return {
            "success": True,
            "target": target_id,
            "topic": topic,
            "old_belief": current_belief,
            "new_belief": new_belief,
            "change": new_belief - current_belief,
        }

    async def _execute_reveal(self, context: "ScenarioContext") -> Dict[str, Any]:
        """Execute information revelation."""
        revealer_id = self._resolve_character(context, self.params.get("revealer"))
        targets = self._resolve_character_list(context)
        information = self.params.get("information", {})

        for target_id in targets:
            target = context.get_character(target_id)
            if not target:
                continue

            # Reveal affects beliefs
            for topic, value in information.items():
                current = target.beliefs.get(topic, 0.5)
                impact = self.params.get("impact", 0.3)
                new_value = current + (value - current) * impact
                target.beliefs[topic] = max(0.0, min(1.0, new_value))

            if revealer_id:
                await context.driver.tell(
                    revealer_id, target_id,
                    f"I need to tell you something important..."
                )

        return {
            "revealer": revealer_id,
            "targets": targets,
            "information": information,
        }


# =============================================================================
# Triggers
# =============================================================================

@dataclass
class Trigger:
    """A trigger that activates based on events."""
    trigger_type: TriggerType
    conditions: Optional[ConditionGroup] = None
    actions: List[ScenarioAction] = field(default_factory=list)
    effects: List[ScenarioEffect] = field(default_factory=list)
    once: bool = False
    enabled: bool = True
    _fired: bool = field(default=False, init=False)

    async def check_and_fire(self, context: "ScenarioContext", event: Dict[str, Any]) -> bool:
        """Check if trigger should fire and execute if so."""
        if not self.enabled or (self.once and self._fired):
            return False

        # Check trigger type matches event
        if event.get("type") != self.trigger_type.value:
            return False

        # Check conditions
        if self.conditions and not self.conditions.evaluate(context):
            return False

        # Fire trigger
        self._fired = True

        # Execute actions
        for action in self.actions:
            await action.execute(context)

        # Apply effects
        for effect in self.effects:
            await effect.apply(context)

        return True


# =============================================================================
# Phases
# =============================================================================

@dataclass
class Phase:
    """A phase in a scenario."""
    name: str
    description: str = ""

    # Entry/exit
    on_enter: List[ScenarioAction] = field(default_factory=list)
    on_exit: List[ScenarioAction] = field(default_factory=list)

    # Main actions
    actions: List[ScenarioAction] = field(default_factory=list)

    # Triggers active during this phase
    triggers: List[Trigger] = field(default_factory=list)

    # Transition conditions
    transitions: Dict[str, ConditionGroup] = field(default_factory=dict)  # phase_name -> condition
    auto_advance: bool = True
    timeout: Optional[float] = None

    # State
    state: PhaseState = PhaseState.PENDING
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None

    async def enter(self, context: "ScenarioContext"):
        """Enter this phase."""
        self.state = PhaseState.ACTIVE
        self.started_at = datetime.now(timezone.utc)
        context.current_phase = self.name
        context.log(f"Entering phase: {self.name}")

        for action in self.on_enter:
            await action.execute(context)

    async def execute(self, context: "ScenarioContext"):
        """Execute the phase's main actions."""
        for action in self.actions:
            if self.state != PhaseState.ACTIVE:
                break
            await action.execute(context)

    async def exit(self, context: "ScenarioContext"):
        """Exit this phase."""
        for action in self.on_exit:
            await action.execute(context)

        self.state = PhaseState.COMPLETED
        self.ended_at = datetime.now(timezone.utc)
        context.log(f"Exiting phase: {self.name}")

    def get_next_phase(self, context: "ScenarioContext") -> Optional[str]:
        """Determine the next phase based on transition conditions."""
        for next_phase, condition in self.transitions.items():
            if condition.evaluate(context):
                return next_phase
        return None


# =============================================================================
# Context
# =============================================================================

@dataclass
class ScenarioContext:
    """Runtime context for scenario execution."""
    scenario_id: str
    driver: SimulationDriver

    # Participants
    participants: Set[str] = field(default_factory=set)
    roles: Dict[str, str] = field(default_factory=dict)  # role -> character_id
    coalitions: Dict[str, Set[str]] = field(default_factory=dict)

    # State
    variables: Dict[str, Any] = field(default_factory=dict)
    current_phase: Optional[str] = None
    events: List[Dict[str, Any]] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    def get_character(self, char_id: str) -> Optional[Character]:
        """Get a character by ID."""
        return self.driver.get_character(char_id)

    def emit_event(self, event_type: str, data: Dict[str, Any] = None):
        """Emit an event."""
        self.events.append({
            "type": event_type,
            "data": data or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def log(self, message: str):
        """Add a log message."""
        self.logs.append(f"[{datetime.now(timezone.utc).isoformat()}] {message}")

    def update_metric(self, name: str, value: Any, operation: str = "set"):
        """Update a metric."""
        if operation == "set":
            self.metrics[name] = value
        elif operation == "increment":
            self.metrics[name] = self.metrics.get(name, 0) + value
        elif operation == "append":
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(value)


# =============================================================================
# Scenario Definition
# =============================================================================

@dataclass
class ScenarioDefinition:
    """Complete definition of a scenario."""
    id: str = field(default_factory=lambda: f"scenario_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""

    # Participants
    required_roles: List[str] = field(default_factory=list)
    min_participants: int = 2
    max_participants: int = 10

    # Initial setup
    initial_variables: Dict[str, Any] = field(default_factory=dict)
    initial_beliefs: Dict[str, Dict[str, float]] = field(default_factory=dict)  # role -> {topic: belief}

    # Phases
    phases: List[Phase] = field(default_factory=list)
    initial_phase: str = ""

    # Global triggers
    triggers: List[Trigger] = field(default_factory=list)

    # End conditions
    success_conditions: Optional[ConditionGroup] = None
    failure_conditions: Optional[ConditionGroup] = None
    max_duration: Optional[float] = None

    # Metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Scenario Executor
# =============================================================================

class ScenarioExecutor:
    """Executes scenario definitions."""

    def __init__(self, driver: SimulationDriver):
        self.driver = driver
        self._running_scenarios: Dict[str, ScenarioContext] = {}

    async def run(
        self,
        definition: ScenarioDefinition,
        role_assignments: Dict[str, str],  # role -> character_id
        extra_participants: List[str] = None,
    ) -> Dict[str, Any]:
        """Run a scenario.

        Args:
            definition: The scenario definition
            role_assignments: Mapping of roles to character IDs
            extra_participants: Additional participants beyond roles

        Returns:
            Scenario results
        """
        # Validate
        for role in definition.required_roles:
            if role not in role_assignments:
                raise ValueError(f"Missing required role: {role}")

        # Create context
        context = ScenarioContext(
            scenario_id=definition.id,
            driver=self.driver,
            participants=set(role_assignments.values()) | set(extra_participants or []),
            roles=role_assignments,
            variables=definition.initial_variables.copy(),
        )

        # Apply initial beliefs
        for role, beliefs in definition.initial_beliefs.items():
            char_id = role_assignments.get(role)
            if char_id:
                char = self.driver.get_character(char_id)
                if char:
                    char.beliefs.update(beliefs)

        self._running_scenarios[definition.id] = context

        # Build phase map
        phase_map = {p.name: p for p in definition.phases}
        current_phase_name = definition.initial_phase or (definition.phases[0].name if definition.phases else None)

        start_time = datetime.now(timezone.utc)
        state = ScenarioState.RUNNING

        try:
            while current_phase_name and state == ScenarioState.RUNNING:
                phase = phase_map.get(current_phase_name)
                if not phase:
                    break

                # Enter phase
                await phase.enter(context)

                # Execute phase with timeout
                if phase.timeout:
                    try:
                        await asyncio.wait_for(phase.execute(context), timeout=phase.timeout)
                    except asyncio.TimeoutError:
                        context.log(f"Phase {phase.name} timed out")
                else:
                    await phase.execute(context)

                # Check phase triggers
                for trigger in phase.triggers:
                    await trigger.check_and_fire(context, {"type": "on_phase_exit"})

                # Exit phase
                await phase.exit(context)

                # Check global triggers
                for trigger in definition.triggers:
                    await trigger.check_and_fire(context, {"type": "on_phase_exit"})

                # Check end conditions
                if definition.success_conditions and definition.success_conditions.evaluate(context):
                    state = ScenarioState.COMPLETED
                    break
                if definition.failure_conditions and definition.failure_conditions.evaluate(context):
                    state = ScenarioState.FAILED
                    break

                # Check max duration
                if definition.max_duration:
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    if elapsed > definition.max_duration:
                        state = ScenarioState.COMPLETED
                        context.log("Max duration reached")
                        break

                # Get next phase
                if phase.auto_advance:
                    current_phase_name = phase.get_next_phase(context)
                    if not current_phase_name and phase.transitions:
                        # Default to first transition if no condition matched
                        current_phase_name = list(phase.transitions.keys())[0] if phase.transitions else None
                    elif not current_phase_name:
                        # Move to next phase in sequence
                        phase_names = [p.name for p in definition.phases]
                        current_idx = phase_names.index(phase.name)
                        if current_idx + 1 < len(phase_names):
                            current_phase_name = phase_names[current_idx + 1]
                        else:
                            current_phase_name = None
                else:
                    current_phase_name = None

            if state == ScenarioState.RUNNING:
                state = ScenarioState.COMPLETED

        except Exception as e:
            state = ScenarioState.FAILED
            context.log(f"Scenario failed: {e}")
            logger.exception("Scenario execution failed")

        finally:
            del self._running_scenarios[definition.id]

        # Build results
        end_time = datetime.now(timezone.utc)
        return {
            "scenario_id": definition.id,
            "scenario_name": definition.name,
            "state": state.value,
            "started_at": start_time.isoformat(),
            "ended_at": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "participants": list(context.participants),
            "roles": context.roles,
            "coalitions": {k: list(v) for k, v in context.coalitions.items()},
            "variables": context.variables,
            "metrics": context.metrics,
            "events": context.events,
            "logs": context.logs,
        }

    async def abort(self, scenario_id: str) -> bool:
        """Abort a running scenario."""
        if scenario_id in self._running_scenarios:
            self._running_scenarios[scenario_id].log("Scenario aborted")
            # In a real implementation, we'd signal the running task
            return True
        return False

    def get_status(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a running scenario."""
        context = self._running_scenarios.get(scenario_id)
        if not context:
            return None

        return {
            "scenario_id": scenario_id,
            "current_phase": context.current_phase,
            "participants": list(context.participants),
            "variables": context.variables,
            "event_count": len(context.events),
        }


# =============================================================================
# Scenario Builder (Fluent API)
# =============================================================================

class ScenarioBuilder:
    """Fluent builder for creating scenarios."""

    def __init__(self, name: str = ""):
        self._definition = ScenarioDefinition(name=name)
        self._current_phase: Optional[Phase] = None

    def with_description(self, description: str) -> "ScenarioBuilder":
        self._definition.description = description
        return self

    def requires_role(self, role: str) -> "ScenarioBuilder":
        self._definition.required_roles.append(role)
        return self

    def with_participants(self, min_p: int, max_p: int) -> "ScenarioBuilder":
        self._definition.min_participants = min_p
        self._definition.max_participants = max_p
        return self

    def with_variable(self, name: str, value: Any) -> "ScenarioBuilder":
        self._definition.initial_variables[name] = value
        return self

    def with_initial_beliefs(self, role: str, beliefs: Dict[str, float]) -> "ScenarioBuilder":
        self._definition.initial_beliefs[role] = beliefs
        return self

    def add_phase(self, name: str, description: str = "") -> "ScenarioBuilder":
        phase = Phase(name=name, description=description)
        self._definition.phases.append(phase)
        self._current_phase = phase
        if not self._definition.initial_phase:
            self._definition.initial_phase = name
        return self

    def on_enter(self, action: ScenarioAction) -> "ScenarioBuilder":
        if self._current_phase:
            self._current_phase.on_enter.append(action)
        return self

    def on_exit(self, action: ScenarioAction) -> "ScenarioBuilder":
        if self._current_phase:
            self._current_phase.on_exit.append(action)
        return self

    def with_action(self, action: ScenarioAction) -> "ScenarioBuilder":
        if self._current_phase:
            self._current_phase.actions.append(action)
        return self

    def with_trigger(self, trigger: Trigger) -> "ScenarioBuilder":
        if self._current_phase:
            self._current_phase.triggers.append(trigger)
        else:
            self._definition.triggers.append(trigger)
        return self

    def transition_to(self, phase_name: str, condition: ConditionGroup = None) -> "ScenarioBuilder":
        if self._current_phase:
            self._current_phase.transitions[phase_name] = condition or ConditionGroup([], mode="all")
        return self

    def with_timeout(self, seconds: float) -> "ScenarioBuilder":
        if self._current_phase:
            self._current_phase.timeout = seconds
        return self

    def success_when(self, condition: ConditionGroup) -> "ScenarioBuilder":
        self._definition.success_conditions = condition
        return self

    def fail_when(self, condition: ConditionGroup) -> "ScenarioBuilder":
        self._definition.failure_conditions = condition
        return self

    def max_duration(self, seconds: float) -> "ScenarioBuilder":
        self._definition.max_duration = seconds
        return self

    def with_tags(self, *tags: str) -> "ScenarioBuilder":
        self._definition.tags.extend(tags)
        return self

    def build(self) -> ScenarioDefinition:
        return self._definition


# =============================================================================
# Scenario Composition
# =============================================================================

class ScenarioComposer:
    """Composes multiple scenarios together."""

    @staticmethod
    def sequence(*scenarios: ScenarioDefinition, name: str = "composed_sequence") -> ScenarioDefinition:
        """Compose scenarios to run in sequence."""
        composed = ScenarioDefinition(name=name)

        for i, scenario in enumerate(scenarios):
            prefix = f"s{i}_"
            for phase in scenario.phases:
                new_phase = Phase(
                    name=f"{prefix}{phase.name}",
                    description=f"[{scenario.name}] {phase.description}",
                    on_enter=phase.on_enter,
                    on_exit=phase.on_exit,
                    actions=phase.actions,
                    triggers=phase.triggers,
                )
                composed.phases.append(new_phase)

            # Merge variables
            for k, v in scenario.initial_variables.items():
                composed.initial_variables[f"{prefix}{k}"] = v

        # Set up transitions
        for i in range(len(composed.phases) - 1):
            composed.phases[i].transitions[composed.phases[i + 1].name] = ConditionGroup([], mode="all")

        if composed.phases:
            composed.initial_phase = composed.phases[0].name

        return composed

    @staticmethod
    def parallel(*scenarios: ScenarioDefinition, name: str = "composed_parallel") -> ScenarioDefinition:
        """Compose scenarios to run in parallel (interleaved)."""
        composed = ScenarioDefinition(name=name)

        # Create a single phase that executes all scenarios' actions
        combined_phase = Phase(name="parallel_execution", description="Parallel execution of multiple scenarios")

        # Interleave actions from all scenarios
        max_phases = max(len(s.phases) for s in scenarios) if scenarios else 0

        for phase_idx in range(max_phases):
            for scenario in scenarios:
                if phase_idx < len(scenario.phases):
                    phase = scenario.phases[phase_idx]
                    combined_phase.actions.extend(phase.actions)

        composed.phases.append(combined_phase)
        composed.initial_phase = "parallel_execution"

        return composed

    @staticmethod
    def branch(
        base: ScenarioDefinition,
        branches: Dict[str, Tuple[ConditionGroup, ScenarioDefinition]],
        name: str = "branched_scenario"
    ) -> ScenarioDefinition:
        """Create a branching scenario based on conditions."""
        composed = ScenarioDefinition(name=name)

        # Add base phases
        for phase in base.phases:
            composed.phases.append(phase)

        # Add branching phase
        branch_phase = Phase(
            name="branch_decision",
            description="Branch decision point",
        )

        for branch_name, (condition, scenario) in branches.items():
            prefix = f"branch_{branch_name}_"
            # Add branch phases
            for phase in scenario.phases:
                new_phase = Phase(
                    name=f"{prefix}{phase.name}",
                    description=phase.description,
                    on_enter=phase.on_enter,
                    on_exit=phase.on_exit,
                    actions=phase.actions,
                    triggers=phase.triggers,
                )
                composed.phases.append(new_phase)

            # Add transition from branch phase
            branch_phase.transitions[f"{prefix}{scenario.phases[0].name}"] = condition

        composed.phases.append(branch_phase)

        # Connect base to branch
        if base.phases:
            base.phases[-1].transitions["branch_decision"] = ConditionGroup([], mode="all")

        composed.initial_phase = base.phases[0].name if base.phases else "branch_decision"

        return composed
