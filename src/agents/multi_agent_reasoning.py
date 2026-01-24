"""Advanced Multi-Agent Reasoning and Coordination.

Sophisticated capabilities for policy wargaming:

1. Theory of Mind - Agents model other agents' beliefs and intentions
2. Strategic Reasoning - Game-theoretic decision making with lookahead
3. Belief Propagation - Information spread and belief updating
4. Emergent Coordination - Self-organizing coalitions and norms
5. Reputation Systems - Trust and credibility tracking
6. Multi-Level Planning - Tactical, operational, strategic, grand strategic
7. Counterfactual Reasoning - "What if" analysis within agents
8. Preference Learning - Infer preferences from observed behavior
9. Communication Protocols - Structured negotiation and signaling
10. Consensus Mechanisms - Multi-party agreement protocols
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Theory of Mind - Mental Models of Other Agents
# =============================================================================


@dataclass
class BeliefState:
    """What an agent believes about the world."""
    beliefs: Dict[str, Any] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)  # 0-1
    last_updated: Dict[str, float] = field(default_factory=dict)

    def update(self, key: str, value: Any, confidence: float = 0.8):
        self.beliefs[key] = value
        self.confidence[key] = confidence
        self.last_updated[key] = time.time()

    def get(self, key: str, default: Any = None) -> Tuple[Any, float]:
        """Get belief and its confidence."""
        return self.beliefs.get(key, default), self.confidence.get(key, 0.0)

    def decay(self, rate: float = 0.01):
        """Decay confidence over time."""
        for key in self.confidence:
            self.confidence[key] = max(0.1, self.confidence[key] - rate)


@dataclass
class IntentionModel:
    """Model of what an agent intends to do."""
    goals: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)
    predicted_actions: List[str] = field(default_factory=list)
    threat_level: float = 0.0  # 0-1, how threatening to self
    cooperation_likelihood: float = 0.5  # 0-1


@dataclass
class MentalModel:
    """Complete mental model of another agent."""
    agent_name: str
    beliefs: BeliefState = field(default_factory=BeliefState)
    intentions: IntentionModel = field(default_factory=IntentionModel)

    # What we think they think about us
    their_model_of_us: Optional["MentalModel"] = None

    # Personality model (inferred)
    risk_tolerance: float = 0.5  # 0=risk averse, 1=risk seeking
    aggression: float = 0.5  # 0=dovish, 1=hawkish
    rationality: float = 0.8  # 0=emotional, 1=perfectly rational
    patience: float = 0.5  # 0=impatient, 1=long time horizon

    # Track record
    prediction_accuracy: float = 0.5  # How well our predictions matched
    interaction_history: List[Dict] = field(default_factory=list)


class TheoryOfMind:
    """Manages mental models of other agents."""

    def __init__(self, self_agent: str):
        self.self_agent = self_agent
        self.models: Dict[str, MentalModel] = {}
        self._prediction_log: List[Dict] = []

    def get_model(self, agent_name: str) -> MentalModel:
        """Get or create mental model of an agent."""
        if agent_name not in self.models:
            self.models[agent_name] = MentalModel(agent_name=agent_name)
        return self.models[agent_name]

    def update_from_action(self, agent_name: str, action: Dict[str, Any]):
        """Update mental model based on observed action."""
        model = self.get_model(agent_name)

        action_type = action.get("type", "")
        target = action.get("target", "")
        content = action.get("content", "")

        # Update aggression estimate
        aggressive_types = ["threat", "ultimatum", "sanction", "escalate"]
        cooperative_types = ["proposal", "concession", "treaty", "deescalate"]

        if action_type in aggressive_types:
            model.aggression = min(1.0, model.aggression + 0.1)
        elif action_type in cooperative_types:
            model.aggression = max(0.0, model.aggression - 0.1)

        # Update beliefs about their intentions
        if target == self.self_agent:
            if action_type in aggressive_types:
                model.intentions.threat_level = min(1.0, model.intentions.threat_level + 0.2)
            elif action_type in cooperative_types:
                model.intentions.cooperation_likelihood = min(1.0, model.intentions.cooperation_likelihood + 0.15)

        # Log for prediction tracking
        model.interaction_history.append({
            "tick": time.time(),
            "action": action_type,
            "target": target,
        })

    def predict_action(
        self,
        agent_name: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Predict what an agent will do next."""
        model = self.get_model(agent_name)

        # Base prediction on personality model
        if model.aggression > 0.7 and model.intentions.threat_level > 0.5:
            predicted_type = "threat" if random.random() < model.aggression else "statement"
        elif model.intentions.cooperation_likelihood > 0.6:
            predicted_type = "proposal" if random.random() < model.intentions.cooperation_likelihood else "statement"
        else:
            predicted_type = "wait"

        prediction = {
            "agent": agent_name,
            "predicted_action": predicted_type,
            "confidence": model.prediction_accuracy,
            "reasoning": f"Based on aggression={model.aggression:.2f}, cooperation={model.intentions.cooperation_likelihood:.2f}",
        }

        self._prediction_log.append({
            "prediction": prediction,
            "timestamp": time.time(),
        })

        return prediction

    def update_prediction_accuracy(self, agent_name: str, actual_action: str):
        """Update accuracy based on whether prediction was correct."""
        model = self.get_model(agent_name)

        # Find most recent prediction for this agent
        recent_predictions = [
            p for p in self._prediction_log
            if p["prediction"]["agent"] == agent_name
        ]

        if recent_predictions:
            last_pred = recent_predictions[-1]["prediction"]
            was_correct = last_pred["predicted_action"] == actual_action

            # Exponential moving average
            alpha = 0.2
            model.prediction_accuracy = (
                alpha * (1.0 if was_correct else 0.0) +
                (1 - alpha) * model.prediction_accuracy
            )

    def what_do_they_think_we_will_do(
        self,
        agent_name: str,
    ) -> Dict[str, Any]:
        """Second-order theory of mind: what do they predict about us?"""
        model = self.get_model(agent_name)

        # Based on how they've been acting toward us
        if model.intentions.threat_level > 0.6:
            # They think we're a threat, so they expect us to be aggressive
            their_prediction = "threat"
        elif model.intentions.cooperation_likelihood > 0.6:
            their_prediction = "cooperate"
        else:
            their_prediction = "uncertain"

        return {
            "their_prediction_about_us": their_prediction,
            "our_confidence": model.prediction_accuracy * 0.7,  # Lower confidence for 2nd order
        }


# =============================================================================
# Strategic Reasoning with Lookahead
# =============================================================================


class StrategicHorizon(Enum):
    """Planning horizons."""
    TACTICAL = 1       # Immediate (1-3 moves)
    OPERATIONAL = 2    # Short-term (3-10 moves)
    STRATEGIC = 3      # Medium-term (10-50 moves)
    GRAND_STRATEGIC = 4  # Long-term (50+ moves)


@dataclass
class StrategicOption:
    """A potential course of action."""
    option_id: str
    description: str
    actions: List[str]
    expected_utility: float = 0.0
    risk: float = 0.0
    time_horizon: StrategicHorizon = StrategicHorizon.TACTICAL

    # Outcomes
    best_case: str = ""
    worst_case: str = ""
    most_likely: str = ""

    # Dependencies
    requires: List[str] = field(default_factory=list)  # Preconditions
    enables: List[str] = field(default_factory=list)   # What this unlocks


@dataclass
class GameState:
    """State for game-theoretic analysis."""
    players: List[str]
    current_player: str
    payoff_matrix: Dict[Tuple[str, ...], Dict[str, float]] = field(default_factory=dict)
    history: List[Tuple[str, str]] = field(default_factory=list)  # (player, action)
    is_terminal: bool = False


class StrategicReasoner:
    """Game-theoretic strategic reasoning."""

    def __init__(self, agent_name: str, theory_of_mind: TheoryOfMind):
        self.agent_name = agent_name
        self.tom = theory_of_mind
        self.options_cache: Dict[str, List[StrategicOption]] = {}

    def generate_options(
        self,
        context: Dict[str, Any],
        horizon: StrategicHorizon = StrategicHorizon.TACTICAL,
    ) -> List[StrategicOption]:
        """Generate strategic options for current situation."""
        options = []

        # Always available options
        options.append(StrategicOption(
            option_id="wait_and_see",
            description="Maintain current posture and observe",
            actions=["wait"],
            expected_utility=0.0,
            risk=0.1,
            time_horizon=horizon,
        ))

        # Context-dependent options
        escalation_level = context.get("escalation_level", 0)
        relationships = context.get("relationships", {})

        if escalation_level < 3:
            options.append(StrategicOption(
                option_id="diplomatic_push",
                description="Pursue diplomatic solution",
                actions=["proposal", "statement"],
                expected_utility=0.3,
                risk=0.2,
                time_horizon=horizon,
                best_case="Agreement reached",
                worst_case="Rejected, minor credibility loss",
            ))

        if escalation_level > 1:
            options.append(StrategicOption(
                option_id="pressure_campaign",
                description="Apply coordinated pressure",
                actions=["threat", "sanction"],
                expected_utility=0.4,
                risk=0.5,
                time_horizon=horizon,
                best_case="Opponent backs down",
                worst_case="Escalation spiral",
            ))

        # Coalition options if we have allies
        allies = [a for a, r in relationships.items() if r in ["ally", "partner"]]
        if allies:
            options.append(StrategicOption(
                option_id="coalition_action",
                description=f"Coordinate with {', '.join(allies[:2])}",
                actions=["alliance", "joint_statement"],
                expected_utility=0.5,
                risk=0.3,
                time_horizon=horizon,
                requires=["allied_support"],
                enables=["collective_bargaining"],
            ))

        return options

    def minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float = float('-inf'),
        beta: float = float('inf'),
        maximizing: bool = True,
    ) -> Tuple[float, str]:
        """Minimax with alpha-beta pruning for two-player scenarios."""
        if depth == 0 or state.is_terminal:
            return self._evaluate_state(state), ""

        if maximizing:
            max_eval = float('-inf')
            best_action = ""
            for action in self._get_available_actions(state):
                new_state = self._apply_action(state, action)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = float('inf')
            best_action = ""
            for action in self._get_available_actions(state):
                new_state = self._apply_action(state, action)
                eval_score, _ = self.minimax(new_state, depth - 1, alpha, beta, True)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def nash_equilibrium_search(
        self,
        players: List[str],
        actions: Dict[str, List[str]],
        payoffs: Callable[[Dict[str, str]], Dict[str, float]],
    ) -> Dict[str, str]:
        """Find approximate Nash equilibrium through iterative best response."""
        # Start with random strategy profile
        strategy = {p: random.choice(actions[p]) for p in players}

        for _ in range(100):  # Max iterations
            changed = False
            for player in players:
                # Find best response to others' strategies
                best_action = None
                best_payoff = float('-inf')

                for action in actions[player]:
                    test_strategy = strategy.copy()
                    test_strategy[player] = action
                    player_payoff = payoffs(test_strategy)[player]

                    if player_payoff > best_payoff:
                        best_payoff = player_payoff
                        best_action = action

                if best_action and best_action != strategy[player]:
                    strategy[player] = best_action
                    changed = True

            if not changed:
                break  # Equilibrium found

        return strategy

    def _evaluate_state(self, state: GameState) -> float:
        """Evaluate game state utility for self."""
        # Simple heuristic based on recent history
        if self.agent_name not in state.payoff_matrix:
            return 0.0

        # Count favorable vs unfavorable outcomes
        favorable = sum(1 for p, a in state.history if p == self.agent_name)
        return favorable * 0.1

    def _get_available_actions(self, state: GameState) -> List[str]:
        """Get available actions in current state."""
        return ["cooperate", "defect", "wait", "propose", "threaten"]

    def _apply_action(self, state: GameState, action: str) -> GameState:
        """Apply action to create new state."""
        new_state = GameState(
            players=state.players,
            current_player=state.players[
                (state.players.index(state.current_player) + 1) % len(state.players)
            ],
            payoff_matrix=state.payoff_matrix.copy(),
            history=state.history + [(state.current_player, action)],
        )
        return new_state

    def evaluate_counterfactual(
        self,
        actual_history: List[Dict],
        counterfactual_action: Dict,
        at_index: int,
    ) -> Dict[str, Any]:
        """Evaluate 'what if' scenario."""
        # Build counterfactual history
        cf_history = actual_history[:at_index]
        cf_history.append(counterfactual_action)

        # Estimate downstream effects
        effects = {
            "escalation_change": 0,
            "relationship_change": {},
            "capability_change": 0,
        }

        action_type = counterfactual_action.get("type", "")
        if action_type in ["threat", "ultimatum"]:
            effects["escalation_change"] = 1
        elif action_type in ["concession", "treaty"]:
            effects["escalation_change"] = -1

        return {
            "counterfactual_action": counterfactual_action,
            "replaced_action": actual_history[at_index] if at_index < len(actual_history) else None,
            "estimated_effects": effects,
            "confidence": 0.5,  # Counterfactuals are inherently uncertain
        }


# =============================================================================
# Belief Propagation and Information Dynamics
# =============================================================================


@dataclass
class InformationItem:
    """A piece of information that can spread."""
    info_id: str
    content: Dict[str, Any]
    source: str
    credibility: float = 1.0
    timestamp: float = field(default_factory=time.time)

    # Spread tracking
    known_by: Set[str] = field(default_factory=set)
    believed_by: Set[str] = field(default_factory=set)


class BeliefPropagation:
    """Models how information and beliefs spread through agent network."""

    def __init__(self):
        self.information: Dict[str, InformationItem] = {}
        self.network: Dict[str, Set[str]] = defaultdict(set)  # agent -> connected agents
        self.trust_matrix: Dict[Tuple[str, str], float] = {}  # (from, to) -> trust

    def add_connection(self, agent_a: str, agent_b: str, bidirectional: bool = True):
        """Add connection between agents."""
        self.network[agent_a].add(agent_b)
        if bidirectional:
            self.network[agent_b].add(agent_a)

    def set_trust(self, from_agent: str, to_agent: str, trust: float):
        """Set trust level from one agent to another."""
        self.trust_matrix[(from_agent, to_agent)] = max(0, min(1, trust))

    def get_trust(self, from_agent: str, to_agent: str) -> float:
        """Get trust level, default 0.5."""
        return self.trust_matrix.get((from_agent, to_agent), 0.5)

    def introduce_information(
        self,
        source: str,
        content: Dict[str, Any],
        credibility: float = 1.0,
    ) -> str:
        """Introduce new information from a source."""
        info_id = f"info_{source}_{int(time.time() * 1000)}"
        self.information[info_id] = InformationItem(
            info_id=info_id,
            content=content,
            source=source,
            credibility=credibility,
            known_by={source},
            believed_by={source} if credibility > 0.5 else set(),
        )
        return info_id

    def propagate_step(self) -> Dict[str, List[str]]:
        """Run one step of belief propagation. Returns who learned what."""
        updates: Dict[str, List[str]] = defaultdict(list)

        for info_id, info in self.information.items():
            # Each agent who knows can spread to connections
            spreaders = list(info.known_by)

            for spreader in spreaders:
                for neighbor in self.network.get(spreader, []):
                    if neighbor not in info.known_by:
                        # Probability of spreading based on:
                        # - Trust in spreader
                        # - Information credibility
                        # - Random chance

                        trust = self.get_trust(neighbor, spreader)
                        spread_prob = trust * info.credibility * 0.3

                        if random.random() < spread_prob:
                            info.known_by.add(neighbor)
                            updates[neighbor].append(info_id)

                            # Decide if they believe it
                            believe_prob = trust * info.credibility
                            if random.random() < believe_prob:
                                info.believed_by.add(neighbor)

        return dict(updates)

    def get_agent_beliefs(self, agent: str) -> List[Dict[str, Any]]:
        """Get all information an agent believes."""
        believed = []
        for info in self.information.values():
            if agent in info.believed_by:
                believed.append({
                    "content": info.content,
                    "source": info.source,
                    "credibility": info.credibility,
                })
        return believed

    def calculate_consensus(self, topic: str) -> Dict[str, Any]:
        """Calculate consensus level on a topic across all agents."""
        relevant_info = [
            info for info in self.information.values()
            if topic in str(info.content)
        ]

        if not relevant_info:
            return {"consensus": 0, "topic": topic, "positions": {}}

        # Count beliefs
        positions: Dict[str, int] = defaultdict(int)
        for info in relevant_info:
            for believer in info.believed_by:
                positions[believer] += 1

        total_agents = len(set().union(*[info.known_by for info in relevant_info]))
        believing = len(set().union(*[info.believed_by for info in relevant_info]))

        return {
            "topic": topic,
            "consensus": believing / total_agents if total_agents > 0 else 0,
            "aware": total_agents,
            "believing": believing,
        }


# =============================================================================
# Reputation and Trust System
# =============================================================================


@dataclass
class ReputationRecord:
    """Record of an agent's reputation."""
    agent_name: str

    # Dimensions
    trustworthiness: float = 0.5  # Keeps promises
    competence: float = 0.5       # Achieves stated goals
    benevolence: float = 0.5      # Acts in others' interests
    integrity: float = 0.5        # Consistent values
    predictability: float = 0.5   # Behaves consistently

    # Track record
    promises_kept: int = 0
    promises_broken: int = 0
    threats_executed: int = 0
    threats_empty: int = 0

    # Reputation events
    events: List[Dict] = field(default_factory=list)

    @property
    def overall(self) -> float:
        """Overall reputation score."""
        return (
            self.trustworthiness * 0.3 +
            self.competence * 0.2 +
            self.benevolence * 0.15 +
            self.integrity * 0.2 +
            self.predictability * 0.15
        )

    def record_event(self, event_type: str, details: str, impact: float):
        """Record a reputation-affecting event."""
        self.events.append({
            "type": event_type,
            "details": details,
            "impact": impact,
            "timestamp": time.time(),
        })

        # Update relevant dimension
        alpha = 0.15  # Learning rate
        if event_type == "promise_kept":
            self.trustworthiness = self.trustworthiness + alpha * (1 - self.trustworthiness)
            self.promises_kept += 1
        elif event_type == "promise_broken":
            self.trustworthiness = self.trustworthiness - alpha * self.trustworthiness
            self.promises_broken += 1
        elif event_type == "threat_executed":
            self.predictability = self.predictability + alpha * (1 - self.predictability)
            self.threats_executed += 1
        elif event_type == "threat_empty":
            self.predictability = self.predictability - alpha * self.predictability
            self.threats_empty += 1
        elif event_type == "goal_achieved":
            self.competence = self.competence + alpha * (1 - self.competence)
        elif event_type == "goal_failed":
            self.competence = self.competence - alpha * self.competence


class ReputationSystem:
    """Manages reputation across all agents."""

    def __init__(self):
        self.records: Dict[str, ReputationRecord] = {}
        self.observers: Dict[str, Set[str]] = defaultdict(set)  # Who observes whom

    def get_record(self, agent: str) -> ReputationRecord:
        if agent not in self.records:
            self.records[agent] = ReputationRecord(agent_name=agent)
        return self.records[agent]

    def record_action(
        self,
        agent: str,
        action_type: str,
        was_promised: bool = False,
        was_threatened: bool = False,
        succeeded: bool = True,
    ):
        """Record an action and update reputation."""
        record = self.get_record(agent)

        if was_promised:
            if succeeded:
                record.record_event("promise_kept", f"Fulfilled {action_type}", 0.1)
            else:
                record.record_event("promise_broken", f"Failed {action_type}", -0.2)

        if was_threatened:
            if succeeded:
                record.record_event("threat_executed", f"Carried out {action_type}", 0.05)
            else:
                record.record_event("threat_empty", f"Did not execute {action_type}", -0.15)

    def get_reputation_from_perspective(
        self,
        observer: str,
        observed: str,
    ) -> Dict[str, float]:
        """Get reputation as seen by a specific observer."""
        record = self.get_record(observed)

        # Base reputation
        rep = {
            "trustworthiness": record.trustworthiness,
            "competence": record.competence,
            "benevolence": record.benevolence,
            "integrity": record.integrity,
            "predictability": record.predictability,
            "overall": record.overall,
        }

        # Adjust based on relationship (allies rate each other higher)
        # This would integrate with relationship system
        return rep

    def get_most_trusted(self, observer: str, candidates: List[str]) -> str:
        """Get most trusted agent from candidates."""
        if not candidates:
            return ""

        return max(
            candidates,
            key=lambda c: self.get_reputation_from_perspective(observer, c)["overall"]
        )


# =============================================================================
# Communication Protocols
# =============================================================================


class MessageType(Enum):
    """Structured message types for agent communication."""
    # Informative
    INFORM = "inform"
    QUERY = "query"
    CONFIRM = "confirm"
    DENY = "deny"

    # Directive
    REQUEST = "request"
    COMMAND = "command"
    PERMIT = "permit"
    FORBID = "forbid"

    # Commissive
    PROMISE = "promise"
    THREAT = "threat"
    OFFER = "offer"
    ACCEPT = "accept"
    REJECT = "reject"
    COUNTER = "counter"

    # Expressive
    PRAISE = "praise"
    CRITICIZE = "criticize"
    APOLOGIZE = "apologize"

    # Declarative
    DECLARE = "declare"
    ANNOUNCE = "announce"


@dataclass
class StructuredMessage:
    """A structured message between agents."""
    message_id: str
    sender: str
    recipients: List[str]
    message_type: MessageType
    content: Dict[str, Any]

    # Conversation tracking
    in_reply_to: Optional[str] = None
    conversation_id: Optional[str] = None

    # Metadata
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 0=normal, 1=urgent, 2=critical
    expires_at: Optional[float] = None

    # Commitments
    creates_commitment: bool = False
    commitment_id: Optional[str] = None


class ConversationManager:
    """Manages multi-turn conversations between agents."""

    def __init__(self):
        self.conversations: Dict[str, List[StructuredMessage]] = {}
        self.active_negotiations: Dict[str, Dict] = {}
        self._msg_counter = 0

    def new_conversation(self, participants: List[str]) -> str:
        """Start a new conversation."""
        conv_id = f"conv_{int(time.time())}_{self._msg_counter}"
        self._msg_counter += 1
        self.conversations[conv_id] = []
        return conv_id

    def add_message(self, msg: StructuredMessage):
        """Add message to conversation."""
        if msg.conversation_id:
            if msg.conversation_id not in self.conversations:
                self.conversations[msg.conversation_id] = []
            self.conversations[msg.conversation_id].append(msg)

    def get_conversation_state(self, conv_id: str) -> Dict[str, Any]:
        """Get current state of a conversation."""
        messages = self.conversations.get(conv_id, [])

        if not messages:
            return {"status": "empty", "messages": 0}

        last_msg = messages[-1]
        pending_responses = set()

        # Find who needs to respond
        for msg in reversed(messages):
            if msg.message_type in [
                MessageType.QUERY, MessageType.REQUEST,
                MessageType.OFFER, MessageType.COUNTER
            ]:
                for recipient in msg.recipients:
                    # Check if they've responded
                    responded = any(
                        m.sender == recipient and m.in_reply_to == msg.message_id
                        for m in messages
                    )
                    if not responded:
                        pending_responses.add(recipient)
                break

        return {
            "status": "active" if pending_responses else "concluded",
            "messages": len(messages),
            "last_type": last_msg.message_type.value,
            "pending_from": list(pending_responses),
        }

    def start_negotiation(
        self,
        initiator: str,
        target: str,
        topic: str,
        initial_offer: Dict[str, Any],
    ) -> str:
        """Start a formal negotiation."""
        conv_id = self.new_conversation([initiator, target])

        self.active_negotiations[conv_id] = {
            "topic": topic,
            "parties": [initiator, target],
            "offers": [{"from": initiator, "offer": initial_offer}],
            "status": "open",
            "started_at": time.time(),
        }

        # Create initial offer message
        msg = StructuredMessage(
            message_id=f"msg_{conv_id}_0",
            sender=initiator,
            recipients=[target],
            message_type=MessageType.OFFER,
            content={"topic": topic, "offer": initial_offer},
            conversation_id=conv_id,
            creates_commitment=True,
        )
        self.add_message(msg)

        return conv_id

    def respond_to_negotiation(
        self,
        conv_id: str,
        responder: str,
        response_type: str,  # "accept", "reject", "counter"
        counter_offer: Optional[Dict[str, Any]] = None,
    ) -> StructuredMessage:
        """Respond to a negotiation."""
        neg = self.active_negotiations.get(conv_id)
        if not neg:
            raise ValueError(f"Negotiation {conv_id} not found")

        messages = self.conversations.get(conv_id, [])
        last_msg = messages[-1] if messages else None

        msg_type_map = {
            "accept": MessageType.ACCEPT,
            "reject": MessageType.REJECT,
            "counter": MessageType.COUNTER,
        }

        msg = StructuredMessage(
            message_id=f"msg_{conv_id}_{len(messages)}",
            sender=responder,
            recipients=[p for p in neg["parties"] if p != responder],
            message_type=msg_type_map.get(response_type, MessageType.REJECT),
            content={
                "response": response_type,
                "counter_offer": counter_offer,
            },
            conversation_id=conv_id,
            in_reply_to=last_msg.message_id if last_msg else None,
            creates_commitment=response_type == "accept",
        )

        self.add_message(msg)

        if response_type == "accept":
            neg["status"] = "accepted"
        elif response_type == "reject":
            neg["status"] = "rejected"
        elif response_type == "counter" and counter_offer:
            neg["offers"].append({"from": responder, "offer": counter_offer})

        return msg


# =============================================================================
# Consensus and Voting Mechanisms
# =============================================================================


class VotingMethod(Enum):
    """Voting methods for multi-party decisions."""
    MAJORITY = "majority"
    SUPERMAJORITY = "supermajority"  # 2/3
    UNANIMITY = "unanimity"
    WEIGHTED = "weighted"
    APPROVAL = "approval"
    RANKED_CHOICE = "ranked_choice"


@dataclass
class Vote:
    """A vote cast by an agent."""
    voter: str
    choice: str
    weight: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Ballot:
    """A ballot for voting."""
    ballot_id: str
    topic: str
    options: List[str]
    voting_method: VotingMethod
    eligible_voters: Set[str]
    deadline: Optional[float] = None

    votes: List[Vote] = field(default_factory=list)
    is_open: bool = True
    result: Optional[str] = None

    def cast_vote(self, voter: str, choice: str, weight: float = 1.0) -> bool:
        """Cast a vote."""
        if not self.is_open:
            return False
        if voter not in self.eligible_voters:
            return False
        if choice not in self.options and self.voting_method != VotingMethod.APPROVAL:
            return False

        # Remove previous vote if exists
        self.votes = [v for v in self.votes if v.voter != voter]

        self.votes.append(Vote(voter=voter, choice=choice, weight=weight))
        return True

    def tally(self) -> Dict[str, float]:
        """Tally votes."""
        counts: Dict[str, float] = defaultdict(float)
        for vote in self.votes:
            counts[vote.choice] += vote.weight
        return dict(counts)

    def close_and_decide(self) -> Optional[str]:
        """Close voting and determine result."""
        self.is_open = False
        counts = self.tally()

        if not counts:
            return None

        total_weight = sum(v.weight for v in self.votes)
        total_eligible = len(self.eligible_voters)

        if self.voting_method == VotingMethod.MAJORITY:
            winner = max(counts.items(), key=lambda x: x[1])
            if winner[1] > total_weight / 2:
                self.result = winner[0]

        elif self.voting_method == VotingMethod.SUPERMAJORITY:
            winner = max(counts.items(), key=lambda x: x[1])
            if winner[1] >= total_weight * 2 / 3:
                self.result = winner[0]

        elif self.voting_method == VotingMethod.UNANIMITY:
            if len(counts) == 1 and len(self.votes) == total_eligible:
                self.result = list(counts.keys())[0]

        elif self.voting_method == VotingMethod.WEIGHTED:
            winner = max(counts.items(), key=lambda x: x[1])
            self.result = winner[0]

        return self.result


class ConsensusBuilder:
    """Builds consensus among multiple agents."""

    def __init__(self):
        self.ballots: Dict[str, Ballot] = {}
        self._ballot_counter = 0

    def create_ballot(
        self,
        topic: str,
        options: List[str],
        voters: Set[str],
        method: VotingMethod = VotingMethod.MAJORITY,
        deadline: Optional[float] = None,
    ) -> str:
        """Create a new ballot."""
        self._ballot_counter += 1
        ballot_id = f"ballot_{self._ballot_counter}"

        self.ballots[ballot_id] = Ballot(
            ballot_id=ballot_id,
            topic=topic,
            options=options,
            voting_method=method,
            eligible_voters=voters,
            deadline=deadline,
        )

        return ballot_id

    def get_ballot(self, ballot_id: str) -> Optional[Ballot]:
        return self.ballots.get(ballot_id)

    def check_consensus_possible(
        self,
        ballot_id: str,
        required_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Check if consensus is still mathematically possible."""
        ballot = self.ballots.get(ballot_id)
        if not ballot:
            return {"possible": False, "reason": "Ballot not found"}

        counts = ballot.tally()
        voted = len(ballot.votes)
        remaining = len(ballot.eligible_voters) - voted

        # Check if any option can still win
        max_possible = {}
        for option in ballot.options:
            current = counts.get(option, 0)
            max_possible[option] = current + remaining

        total = voted + remaining
        threshold = total * required_threshold

        achievable = [opt for opt, max_val in max_possible.items() if max_val >= threshold]

        return {
            "possible": len(achievable) > 0,
            "achievable_options": achievable,
            "votes_remaining": remaining,
            "current_leader": max(counts.items(), key=lambda x: x[1])[0] if counts else None,
        }


# =============================================================================
# Multi-Agent Coordination Framework
# =============================================================================


class CoordinationProtocol(Enum):
    """Coordination protocols."""
    CONTRACT_NET = "contract_net"  # Task allocation
    AUCTION = "auction"            # Resource allocation
    VOTING = "voting"              # Decision making
    NEGOTIATION = "negotiation"    # Bilateral/multilateral deals
    ARGUMENTATION = "argumentation"  # Reasoned persuasion


@dataclass
class CoordinationSession:
    """A coordination session between multiple agents."""
    session_id: str
    protocol: CoordinationProtocol
    participants: Set[str]
    coordinator: Optional[str] = None
    topic: str = ""

    # State
    phase: str = "init"
    started_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None

    # Results
    outcome: Optional[Dict[str, Any]] = None
    is_complete: bool = False


class MultiAgentCoordinator:
    """Coordinates complex multi-agent interactions."""

    def __init__(self):
        self.sessions: Dict[str, CoordinationSession] = {}
        self.conversation_mgr = ConversationManager()
        self.consensus_builder = ConsensusBuilder()
        self.belief_propagation = BeliefPropagation()
        self.reputation_system = ReputationSystem()

        self._session_counter = 0

    def create_session(
        self,
        protocol: CoordinationProtocol,
        participants: Set[str],
        topic: str,
        coordinator: Optional[str] = None,
        deadline: Optional[float] = None,
    ) -> str:
        """Create a new coordination session."""
        self._session_counter += 1
        session_id = f"session_{self._session_counter}"

        self.sessions[session_id] = CoordinationSession(
            session_id=session_id,
            protocol=protocol,
            participants=participants,
            coordinator=coordinator or next(iter(participants), None),
            topic=topic,
            deadline=deadline,
        )

        # Setup based on protocol
        if protocol == CoordinationProtocol.VOTING:
            # Create initial ballot
            pass
        elif protocol == CoordinationProtocol.NEGOTIATION:
            # Start conversation
            self.conversation_mgr.new_conversation(list(participants))

        return session_id

    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a coordination session."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "session_id": session_id,
            "protocol": session.protocol.value,
            "participants": list(session.participants),
            "phase": session.phase,
            "is_complete": session.is_complete,
            "outcome": session.outcome,
        }

    def broadcast_to_session(
        self,
        session_id: str,
        sender: str,
        message_type: MessageType,
        content: Dict[str, Any],
    ) -> str:
        """Send a message to all session participants."""
        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        recipients = [p for p in session.participants if p != sender]

        msg = StructuredMessage(
            message_id=f"msg_{session_id}_{int(time.time())}",
            sender=sender,
            recipients=recipients,
            message_type=message_type,
            content=content,
        )

        return msg.message_id

    def run_contract_net(
        self,
        session_id: str,
        task: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Contract Net Protocol for task allocation."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        session.phase = "announce"

        # Phase 1: Announce task (simulated)
        bids: Dict[str, Dict] = {}

        # Phase 2: Collect bids (simulated based on reputation)
        for participant in session.participants:
            if participant != session.coordinator:
                rep = self.reputation_system.get_record(participant)
                # Generate simulated bid based on competence
                bid = {
                    "bidder": participant,
                    "price": random.uniform(0.5, 1.5) * (2 - rep.competence),
                    "quality": rep.competence,
                    "timeline": random.randint(1, 10),
                }
                bids[participant] = bid

        session.phase = "evaluate"

        # Phase 3: Select winner
        if bids:
            winner = min(
                bids.items(),
                key=lambda x: x[1]["price"] / (x[1]["quality"] + 0.1)
            )
            session.outcome = {
                "winner": winner[0],
                "bid": winner[1],
                "task": task,
            }

        session.phase = "complete"
        session.is_complete = True

        return session.outcome or {}

    def run_auction(
        self,
        session_id: str,
        item: Dict[str, Any],
        auction_type: str = "english",  # english, dutch, sealed_bid
    ) -> Dict[str, Any]:
        """Run an auction protocol."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        session.phase = "bidding"

        bids: List[Tuple[str, float]] = []

        # Simulate bids
        for participant in session.participants:
            if participant != session.coordinator:
                rep = self.reputation_system.get_record(participant)
                # Bid based on perceived value and risk tolerance
                base_value = item.get("reserve_price", 100)
                bid_amount = base_value * random.uniform(0.8, 1.5)
                bids.append((participant, bid_amount))

        session.phase = "complete"
        session.is_complete = True

        if bids:
            winner = max(bids, key=lambda x: x[1])
            session.outcome = {
                "winner": winner[0],
                "winning_bid": winner[1],
                "item": item,
            }

        return session.outcome or {}


# =============================================================================
# Integration: Enhanced Agent with All Capabilities
# =============================================================================


class EnhancedReasoningAgent:
    """An agent with full multi-agent reasoning capabilities."""

    def __init__(
        self,
        name: str,
        coordinator: MultiAgentCoordinator,
    ):
        self.name = name
        self.coordinator = coordinator

        # Reasoning capabilities
        self.theory_of_mind = TheoryOfMind(name)
        self.strategic_reasoner = StrategicReasoner(name, self.theory_of_mind)

        # State
        self.beliefs = BeliefState()
        self.active_commitments: List[str] = []
        self.active_sessions: List[str] = []

    def observe_action(self, actor: str, action: Dict[str, Any]):
        """Process an observed action."""
        # Update theory of mind
        self.theory_of_mind.update_from_action(actor, action)

        # Update beliefs
        self.beliefs.update(
            f"last_action_{actor}",
            action,
            confidence=0.9
        )

    def decide_action(
        self,
        context: Dict[str, Any],
        available_actions: List[str],
    ) -> Dict[str, Any]:
        """Decide on an action using strategic reasoning."""
        # Generate options
        options = self.strategic_reasoner.generate_options(context)

        # Predict others' responses
        predictions = {}
        for other in context.get("other_agents", []):
            predictions[other] = self.theory_of_mind.predict_action(other, context)

        # Select best option considering predictions
        best_option = max(
            options,
            key=lambda o: self._evaluate_option(o, predictions, context)
        )

        return {
            "action": best_option.actions[0] if best_option.actions else "wait",
            "option": best_option,
            "predictions": predictions,
            "reasoning": f"Selected {best_option.option_id} with expected utility {best_option.expected_utility}",
        }

    def _evaluate_option(
        self,
        option: StrategicOption,
        predictions: Dict[str, Dict],
        context: Dict[str, Any],
    ) -> float:
        """Evaluate an option considering predicted responses."""
        base_utility = option.expected_utility
        risk_penalty = option.risk * 0.5

        # Adjust based on predicted responses
        response_adjustment = 0.0
        for agent, prediction in predictions.items():
            if prediction.get("predicted_action") in ["threat", "escalate"]:
                response_adjustment -= 0.2
            elif prediction.get("predicted_action") in ["cooperate", "accept"]:
                response_adjustment += 0.1

        return base_utility - risk_penalty + response_adjustment

    def participate_in_negotiation(
        self,
        session_id: str,
        their_offer: Dict[str, Any],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Participate in a negotiation session."""
        session = self.coordinator.sessions.get(session_id)
        if not session:
            return "reject", None

        # Evaluate offer using theory of mind
        other_party = next(
            (p for p in session.participants if p != self.name),
            None
        )

        if other_party:
            their_model = self.theory_of_mind.get_model(other_party)

            # Calculate acceptable threshold based on their desperation
            their_patience = their_model.patience
            our_threshold = 0.4 + (1 - their_patience) * 0.3

            offer_value = their_offer.get("value", 0.5)

            if offer_value >= our_threshold:
                return "accept", None
            elif offer_value >= our_threshold * 0.7:
                # Counter-offer
                counter = their_offer.copy()
                counter["value"] = (offer_value + our_threshold) / 2
                return "counter", counter
            else:
                return "reject", None

        return "reject", None

    def update_from_outcome(self, outcome: Dict[str, Any]):
        """Update internal models based on outcome."""
        # Update beliefs
        self.beliefs.update("last_outcome", outcome)

        # Update reputation observations
        for agent, result in outcome.get("agent_results", {}).items():
            if agent != self.name:
                was_promise = outcome.get("was_commitment", False)
                succeeded = result.get("succeeded", True)
                self.coordinator.reputation_system.record_action(
                    agent,
                    outcome.get("action_type", "unknown"),
                    was_promised=was_promise,
                    succeeded=succeeded,
                )
