"""Coalition Dynamics and Multi-Party Coordination for Policy Simulations.

Advanced features for multi-agent policy wargaming:

1. Coalition Formation - Agents form/dissolve alliances
2. Power Balancing - Agents react to power shifts
3. Bandwagoning vs Balancing - Strategic alignment choices
4. Commitment Problems - Credibility and follow-through
5. Information Asymmetry - Private beliefs and signaling
6. Audience Costs - Domestic political constraints
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

from .simulation_engine import (
    Action,
    ActionType,
    AgentState,
    EscalationLevel,
    Event,
    Persona,
    RelationshipType,
    SimulationAgent,
    SimulationEngine,
    WorldState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Power and Influence Metrics
# =============================================================================


@dataclass
class PowerMetrics:
    """Quantified power metrics for an agent."""
    agent_name: str

    # Hard power
    military_capability: float = 0.5  # 0-1
    economic_capability: float = 0.5  # 0-1
    technological_capability: float = 0.5  # 0-1

    # Soft power
    diplomatic_influence: float = 0.5  # 0-1
    institutional_leverage: float = 0.5  # 0-1
    narrative_control: float = 0.5  # 0-1

    # Computed
    @property
    def hard_power(self) -> float:
        return (self.military_capability + self.economic_capability + self.technological_capability) / 3

    @property
    def soft_power(self) -> float:
        return (self.diplomatic_influence + self.institutional_leverage + self.narrative_control) / 3

    @property
    def total_power(self) -> float:
        return 0.6 * self.hard_power + 0.4 * self.soft_power


class PowerCalculator:
    """Calculates power metrics from agent properties and world state."""

    def __init__(self, world: WorldState):
        self.world = world

    def calculate_for_agent(self, agent: SimulationAgent) -> PowerMetrics:
        """Calculate power metrics for an agent."""
        persona = agent.persona
        category = persona.category.lower()

        metrics = PowerMetrics(agent_name=agent.name)

        # Category-based baseline
        if "political" in category:
            if "president" in persona.current_role.lower():
                metrics.military_capability = 0.9
                metrics.economic_capability = 0.8
                metrics.diplomatic_influence = 0.85
            elif "senator" in persona.current_role.lower():
                metrics.institutional_leverage = 0.7
                metrics.narrative_control = 0.5
            elif "secretary" in persona.current_role.lower():
                metrics.diplomatic_influence = 0.75
                metrics.institutional_leverage = 0.6

        elif "tech" in category or "ai" in category.lower():
            metrics.technological_capability = 0.85
            metrics.economic_capability = 0.7
            metrics.narrative_control = 0.6

        # Adjust for world state
        if "china" in agent.name.lower() or "xi" in agent.name.lower():
            metrics.military_capability *= (self.world.compute.china_share / 0.25)
            metrics.economic_capability *= 0.9
            metrics.technological_capability *= (self.world.compute.china_share / 0.25)

        if any(x in agent.name.lower() for x in ["raimondo", "sullivan", "biden"]):
            metrics.military_capability *= (self.world.compute.us_share / 0.35)
            metrics.institutional_leverage = 0.8

        # Adjust for agent state
        state = agent.state
        metrics.diplomatic_influence *= state.credibility
        metrics.narrative_control *= state.political_capital

        return metrics


# =============================================================================
# Coalition Structures
# =============================================================================


class CoalitionType(str, Enum):
    """Types of coalitions."""
    FORMAL_ALLIANCE = "formal_alliance"  # Treaty-bound
    STRATEGIC_PARTNERSHIP = "strategic_partnership"  # Flexible alignment
    TACTICAL_COOPERATION = "tactical_cooperation"  # Issue-specific
    BALANCING_COALITION = "balancing_coalition"  # Against a threat
    BANDWAGONING = "bandwagoning"  # Joining the stronger side


@dataclass
class Coalition:
    """A coalition of agents."""
    coalition_id: str
    name: str
    coalition_type: CoalitionType
    members: Set[str]  # Agent names
    leader: Optional[str] = None

    # Formation
    formed_at_tick: int = 0
    formed_against: Optional[str] = None  # For balancing coalitions

    # Commitments
    mutual_defense: bool = False
    technology_sharing: bool = False
    economic_integration: bool = False

    # State
    cohesion: float = 1.0  # 0-1, how unified
    is_active: bool = True

    def combined_power(self, power_map: Dict[str, PowerMetrics]) -> float:
        """Calculate combined power of coalition members."""
        return sum(
            power_map[m].total_power
            for m in self.members
            if m in power_map
        )

    def add_member(self, agent_name: str):
        self.members.add(agent_name)

    def remove_member(self, agent_name: str):
        self.members.discard(agent_name)
        if agent_name == self.leader:
            self.leader = next(iter(self.members), None)


@dataclass
class CoalitionProposal:
    """A proposal to form or modify a coalition."""
    proposal_id: str
    proposer: str
    proposal_type: str  # "form", "join", "leave", "dissolve"
    target_coalition: Optional[str] = None
    invited_members: Set[str] = field(default_factory=set)
    coalition_type: CoalitionType = CoalitionType.STRATEGIC_PARTNERSHIP
    terms: Dict[str, Any] = field(default_factory=dict)

    # Responses
    responses: Dict[str, str] = field(default_factory=dict)  # agent -> accept/reject

    def all_accepted(self) -> bool:
        return all(r == "accept" for r in self.responses.values())


# =============================================================================
# Coalition Manager
# =============================================================================


class CoalitionManager:
    """Manages coalition formation, maintenance, and dissolution."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.coalitions: Dict[str, Coalition] = {}
        self.proposals: Dict[str, CoalitionProposal] = {}
        self.power_map: Dict[str, PowerMetrics] = {}
        self._proposal_counter = 0

    def update_power_metrics(self):
        """Update power metrics for all agents."""
        calculator = PowerCalculator(self.engine.world)
        for agent in self.engine.agents.values():
            self.power_map[agent.name] = calculator.calculate_for_agent(agent)

    def get_agent_coalitions(self, agent_name: str) -> List[Coalition]:
        """Get all coalitions an agent belongs to."""
        return [
            c for c in self.coalitions.values()
            if agent_name in c.members and c.is_active
        ]

    def get_coalition_balance(self) -> Dict[str, float]:
        """Get power balance between coalitions."""
        self.update_power_metrics()
        balance = {}
        for cid, coalition in self.coalitions.items():
            if coalition.is_active:
                balance[cid] = coalition.combined_power(self.power_map)
        return balance

    async def propose_coalition(
        self,
        proposer: str,
        invited: Set[str],
        coalition_type: CoalitionType,
        name: str,
        terms: Optional[Dict[str, Any]] = None,
    ) -> CoalitionProposal:
        """Propose a new coalition."""
        self._proposal_counter += 1
        proposal = CoalitionProposal(
            proposal_id=f"proposal_{self._proposal_counter}",
            proposer=proposer,
            proposal_type="form",
            invited_members=invited,
            coalition_type=coalition_type,
            terms=terms or {},
        )
        self.proposals[proposal.proposal_id] = proposal

        # Create event
        event = Event(
            event_id=f"coalition_proposal_{proposal.proposal_id}",
            event_type="coalition_proposal",
            source=proposer,
            target=None,
            description=f"{proposer} proposes a {coalition_type.value} coalition '{name}' with {', '.join(invited)}",
            tick=self.engine.world.tick,
            visibility="public",
        )
        self.engine.inject_event(event)

        return proposal

    async def respond_to_proposal(
        self,
        proposal_id: str,
        responder: str,
        accept: bool,
    ):
        """Respond to a coalition proposal."""
        if proposal_id not in self.proposals:
            return

        proposal = self.proposals[proposal_id]

        if responder not in proposal.invited_members:
            return

        proposal.responses[responder] = "accept" if accept else "reject"

        # Check if all have responded
        if len(proposal.responses) == len(proposal.invited_members):
            if proposal.all_accepted():
                await self._form_coalition(proposal)
            else:
                self._reject_proposal(proposal)

    async def _form_coalition(self, proposal: CoalitionProposal):
        """Form a coalition from a successful proposal."""
        coalition = Coalition(
            coalition_id=f"coalition_{len(self.coalitions)}",
            name=proposal.terms.get("name", f"Coalition of {proposal.proposer}"),
            coalition_type=proposal.coalition_type,
            members={proposal.proposer} | proposal.invited_members,
            leader=proposal.proposer,
            formed_at_tick=self.engine.world.tick,
            mutual_defense=proposal.terms.get("mutual_defense", False),
            technology_sharing=proposal.terms.get("technology_sharing", False),
            economic_integration=proposal.terms.get("economic_integration", False),
        )
        self.coalitions[coalition.coalition_id] = coalition

        # Create event
        event = Event(
            event_id=f"coalition_formed_{coalition.coalition_id}",
            event_type="coalition_formed",
            source=proposal.proposer,
            target=None,
            description=f"Coalition '{coalition.name}' formed with members: {', '.join(coalition.members)}",
            tick=self.engine.world.tick,
            visibility="public",
            effects={"escalation_delta": -1 if coalition.mutual_defense else 0},
        )
        self.engine.inject_event(event)

    def _reject_proposal(self, proposal: CoalitionProposal):
        """Handle rejected proposal."""
        rejecters = [a for a, r in proposal.responses.items() if r == "reject"]
        event = Event(
            event_id=f"proposal_rejected_{proposal.proposal_id}",
            event_type="coalition_rejected",
            source=proposal.proposer,
            target=None,
            description=f"Coalition proposal rejected by: {', '.join(rejecters)}",
            tick=self.engine.world.tick,
        )
        self.engine.inject_event(event)

    def calculate_balancing_need(self, agent_name: str) -> Optional[str]:
        """Determine if agent should balance against a rising power."""
        self.update_power_metrics()

        agent_power = self.power_map.get(agent_name, PowerMetrics(agent_name)).total_power

        # Find most powerful agent
        max_power = 0
        hegemon = None
        for name, metrics in self.power_map.items():
            if name != agent_name and metrics.total_power > max_power:
                max_power = metrics.total_power
                hegemon = name

        # Check if hegemon is much more powerful
        if hegemon and max_power > agent_power * 1.5:
            return hegemon

        return None

    def should_bandwagon(self, agent_name: str, rising_power: str) -> bool:
        """Determine if agent should bandwagon with rising power."""
        agent = self.engine.agents.get(agent_name)
        if not agent:
            return False

        relationship = agent.get_relationship_with(rising_power)

        # More likely to bandwagon if:
        # - Already allied/partnered
        # - Weak relative to rising power
        # - Rising power's victory seems inevitable
        if relationship in [RelationshipType.ALLY, RelationshipType.PARTNER]:
            return True

        agent_power = self.power_map.get(agent_name, PowerMetrics(agent_name)).total_power
        rising_power_metrics = self.power_map.get(rising_power, PowerMetrics(rising_power))

        if rising_power_metrics.total_power > agent_power * 2:
            return random.random() < 0.6  # 60% chance to bandwagon

        return False

    def update_cohesion(self, coalition_id: str, delta: float):
        """Update coalition cohesion."""
        if coalition_id in self.coalitions:
            self.coalitions[coalition_id].cohesion = max(
                0,
                min(1, self.coalitions[coalition_id].cohesion + delta)
            )

            # Dissolve if cohesion too low
            if self.coalitions[coalition_id].cohesion < 0.2:
                self._dissolve_coalition(coalition_id)

    def _dissolve_coalition(self, coalition_id: str):
        """Dissolve a coalition."""
        if coalition_id not in self.coalitions:
            return

        coalition = self.coalitions[coalition_id]
        coalition.is_active = False

        event = Event(
            event_id=f"coalition_dissolved_{coalition_id}",
            event_type="coalition_dissolved",
            source="system",
            target=None,
            description=f"Coalition '{coalition.name}' has dissolved due to internal disagreements.",
            tick=self.engine.world.tick,
            effects={"escalation_delta": 1},
        )
        self.engine.inject_event(event)


# =============================================================================
# Commitment and Credibility
# =============================================================================


@dataclass
class Commitment:
    """A commitment made by an agent."""
    commitment_id: str
    agent: str
    commitment_type: str  # "promise", "threat", "red_line"
    content: str
    target: Optional[str] = None
    made_at_tick: int = 0
    deadline_tick: Optional[int] = None

    # State
    is_active: bool = True
    was_honored: Optional[bool] = None
    was_tested: bool = False


class CommitmentTracker:
    """Tracks commitments and their credibility implications."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.commitments: Dict[str, Commitment] = {}
        self.credibility_history: Dict[str, List[Dict]] = {}  # agent -> history

    def record_commitment(
        self,
        agent: str,
        commitment_type: str,
        content: str,
        target: Optional[str] = None,
        deadline: Optional[int] = None,
    ) -> Commitment:
        """Record a new commitment."""
        cid = f"commit_{agent}_{self.engine.world.tick}"
        commitment = Commitment(
            commitment_id=cid,
            agent=agent,
            commitment_type=commitment_type,
            content=content,
            target=target,
            made_at_tick=self.engine.world.tick,
            deadline_tick=deadline,
        )
        self.commitments[cid] = commitment

        # Add to agent state
        if agent in self.engine.agents:
            if commitment_type == "red_line":
                self.engine.agents[agent].state.stated_red_lines.append(content)
            else:
                self.engine.agents[agent].state.active_commitments.append(content)

        return commitment

    def test_commitment(self, commitment_id: str, honored: bool):
        """Test whether a commitment was honored."""
        if commitment_id not in self.commitments:
            return

        commitment = self.commitments[commitment_id]
        commitment.was_tested = True
        commitment.was_honored = honored
        commitment.is_active = False

        # Update credibility
        agent = commitment.agent
        if agent not in self.credibility_history:
            self.credibility_history[agent] = []

        self.credibility_history[agent].append({
            "tick": self.engine.world.tick,
            "commitment_id": commitment_id,
            "type": commitment.commitment_type,
            "honored": honored,
        })

        # Update agent credibility
        if agent in self.engine.agents:
            delta = 0.1 if honored else -0.2
            self.engine.agents[agent].state.credibility = max(
                0,
                min(1, self.engine.agents[agent].state.credibility + delta)
            )

    def get_agent_credibility_score(self, agent: str) -> float:
        """Calculate credibility score from history."""
        history = self.credibility_history.get(agent, [])
        if not history:
            return 1.0  # Benefit of doubt

        honored = sum(1 for h in history if h["honored"])
        total = len(history)

        return honored / total

    def get_red_line_violations(self, agent: str) -> List[Commitment]:
        """Get red lines that were violated by others."""
        return [
            c for c in self.commitments.values()
            if c.agent == agent and c.commitment_type == "red_line" and c.was_tested and not c.was_honored
        ]


# =============================================================================
# Audience Costs
# =============================================================================


@dataclass
class DomesticAudience:
    """Represents domestic political constraints."""
    agent_name: str

    # Constituencies
    hawk_faction: float = 0.3  # Support for tough stance
    dove_faction: float = 0.3  # Support for conciliation
    neutral_faction: float = 0.4

    # Current support
    approval_rating: float = 0.5

    # Sensitivity to different outcomes
    backing_down_cost: float = 0.3  # Cost of backing down
    escalation_cost: float = 0.2  # Cost of escalating

    def calculate_audience_cost(self, action: Action) -> float:
        """Calculate audience cost of an action."""
        if action.action_type in [ActionType.CONCESSION, ActionType.DEESCALATE]:
            # Backing down costs hawk support
            return self.backing_down_cost * self.hawk_faction

        elif action.action_type in [ActionType.THREAT, ActionType.ULTIMATUM, ActionType.ESCALATE]:
            # Escalating costs dove support
            return self.escalation_cost * self.dove_faction

        return 0.0

    def update_factions(self, world_escalation: EscalationLevel):
        """Update faction strengths based on world state."""
        if world_escalation.value >= EscalationLevel.CRISIS.value:
            # Crisis rallies hawks
            self.hawk_faction = min(0.5, self.hawk_faction + 0.05)
            self.neutral_faction = max(0.2, self.neutral_faction - 0.03)

        elif world_escalation.value <= EscalationLevel.TENSION.value:
            # Peace favors doves
            self.dove_faction = min(0.5, self.dove_faction + 0.05)
            self.neutral_faction = max(0.2, self.neutral_faction - 0.03)


class AudienceCostManager:
    """Manages audience costs for political agents."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.audiences: Dict[str, DomesticAudience] = {}

    def initialize_audience(self, agent_name: str, **kwargs):
        """Initialize domestic audience for an agent."""
        self.audiences[agent_name] = DomesticAudience(agent_name, **kwargs)

    def get_action_cost(self, action: Action) -> float:
        """Get the audience cost of an action."""
        if action.agent_name not in self.audiences:
            return 0.0

        audience = self.audiences[action.agent_name]
        return audience.calculate_audience_cost(action)

    def apply_action_cost(self, action: Action):
        """Apply audience cost to agent state."""
        cost = self.get_action_cost(action)

        if cost > 0 and action.agent_name in self.engine.agents:
            agent = self.engine.agents[action.agent_name]
            agent.state.political_capital = max(0, agent.state.political_capital - cost)

    def update_all_audiences(self):
        """Update all audiences based on world state."""
        for audience in self.audiences.values():
            audience.update_factions(self.engine.world.geopolitics.escalation_level)


# =============================================================================
# Information Asymmetry and Signaling
# =============================================================================


@dataclass
class PrivateInformation:
    """Private information held by an agent."""
    info_id: str
    holder: str
    info_type: str  # "capability", "intention", "intelligence"
    content: Dict[str, Any]
    true_value: Any
    revealed_to: Set[str] = field(default_factory=set)


@dataclass
class Signal:
    """A signal sent by an agent."""
    signal_id: str
    sender: str
    signal_type: str  # "costly", "cheap_talk", "demonstration"
    content: str
    cost_incurred: float = 0.0
    intended_message: str = ""


class InformationManager:
    """Manages information asymmetry and signaling."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine
        self.private_info: Dict[str, List[PrivateInformation]] = {}
        self.signals: List[Signal] = []

    def add_private_info(
        self,
        holder: str,
        info_type: str,
        content: Dict[str, Any],
        true_value: Any,
    ) -> PrivateInformation:
        """Add private information for an agent."""
        info = PrivateInformation(
            info_id=f"info_{holder}_{len(self.private_info.get(holder, []))}",
            holder=holder,
            info_type=info_type,
            content=content,
            true_value=true_value,
        )

        if holder not in self.private_info:
            self.private_info[holder] = []
        self.private_info[holder].append(info)

        # Store in agent's private beliefs
        if holder in self.engine.agents:
            self.engine.agents[holder].state.private_beliefs[info_type] = true_value

        return info

    def reveal_information(
        self,
        info_id: str,
        to_agents: Set[str],
    ):
        """Reveal private information to other agents."""
        for holder_info in self.private_info.values():
            for info in holder_info:
                if info.info_id == info_id:
                    info.revealed_to.update(to_agents)

                    event = Event(
                        event_id=f"info_revealed_{info_id}",
                        event_type="information_revealed",
                        source=info.holder,
                        target=None,
                        description=f"Information about {info.info_type} revealed to {', '.join(to_agents)}",
                        tick=self.engine.world.tick,
                        visibility="private",
                    )
                    self.engine.inject_event(event)
                    return

    def send_costly_signal(
        self,
        sender: str,
        signal_content: str,
        cost: float,
        intended_message: str,
    ) -> Signal:
        """Send a costly signal (credible because expensive)."""
        signal = Signal(
            signal_id=f"signal_{sender}_{len(self.signals)}",
            sender=sender,
            signal_type="costly",
            content=signal_content,
            cost_incurred=cost,
            intended_message=intended_message,
        )
        self.signals.append(signal)

        # Apply cost to sender
        if sender in self.engine.agents:
            agent = self.engine.agents[sender]
            agent.state.political_capital = max(0, agent.state.political_capital - cost)

        event = Event(
            event_id=f"costly_signal_{signal.signal_id}",
            event_type="costly_signal",
            source=sender,
            target=None,
            description=signal_content,
            tick=self.engine.world.tick,
        )
        self.engine.inject_event(event)

        return signal

    def get_agent_knowledge(self, agent_name: str) -> Dict[str, Any]:
        """Get all information known to an agent."""
        knowledge = {}

        for holder, info_list in self.private_info.items():
            if holder == agent_name:
                # Own private info
                for info in info_list:
                    knowledge[f"own:{info.info_type}"] = info.true_value
            else:
                # Info revealed to this agent
                for info in info_list:
                    if agent_name in info.revealed_to:
                        knowledge[f"{holder}:{info.info_type}"] = info.true_value

        return knowledge


# =============================================================================
# Strategic Interaction Analysis
# =============================================================================


class GameTheoreticAnalyzer:
    """Analyze strategic interactions between agents."""

    def __init__(self, engine: SimulationEngine):
        self.engine = engine

    def analyze_chicken_game(
        self,
        agent_a: str,
        agent_b: str,
        stake: str,
    ) -> Dict[str, Any]:
        """Analyze a chicken game scenario."""
        a_state = self.engine.agents[agent_a].state
        b_state = self.engine.agents[agent_b].state

        # Calculate who is more likely to back down
        a_resolve = (1 - a_state.stress_level) * a_state.credibility
        b_resolve = (1 - b_state.stress_level) * b_state.credibility

        # Consider audience costs
        # Agents with more political capital have more to lose

        analysis = {
            "stake": stake,
            "agent_a": {
                "name": agent_a,
                "resolve": a_resolve,
                "stress": a_state.stress_level,
                "credibility": a_state.credibility,
            },
            "agent_b": {
                "name": agent_b,
                "resolve": b_resolve,
                "stress": b_state.stress_level,
                "credibility": b_state.credibility,
            },
            "predicted_outcome": "a_wins" if a_resolve > b_resolve else "b_wins" if b_resolve > a_resolve else "mutual_destruction_risk",
            "mutual_destruction_probability": min(a_resolve, b_resolve),
        }

        return analysis

    def analyze_prisoners_dilemma(
        self,
        agent_a: str,
        agent_b: str,
        cooperation_payoff: float = 3,
        defection_payoff: float = 5,
        sucker_payoff: float = 0,
        mutual_defection: float = 1,
    ) -> Dict[str, Any]:
        """Analyze cooperation likelihood in prisoner's dilemma."""
        relationship_a_to_b = self.engine.agents[agent_a].get_relationship_with(agent_b)
        relationship_b_to_a = self.engine.agents[agent_b].get_relationship_with(agent_a)

        # Cooperation more likely with:
        # - Good relationship
        # - High credibility (can commit)
        # - Repeated interaction expected

        a_coop_tendency = 0.5
        b_coop_tendency = 0.5

        if relationship_a_to_b in [RelationshipType.ALLY, RelationshipType.PARTNER]:
            a_coop_tendency += 0.3
        elif relationship_a_to_b in [RelationshipType.ADVERSARY, RelationshipType.ENEMY]:
            a_coop_tendency -= 0.3

        if relationship_b_to_a in [RelationshipType.ALLY, RelationshipType.PARTNER]:
            b_coop_tendency += 0.3
        elif relationship_b_to_a in [RelationshipType.ADVERSARY, RelationshipType.ENEMY]:
            b_coop_tendency -= 0.3

        return {
            "agent_a_cooperate_probability": max(0, min(1, a_coop_tendency)),
            "agent_b_cooperate_probability": max(0, min(1, b_coop_tendency)),
            "mutual_cooperation_probability": max(0, min(1, a_coop_tendency)) * max(0, min(1, b_coop_tendency)),
            "payoff_matrix": {
                "both_cooperate": (cooperation_payoff, cooperation_payoff),
                "a_defects": (defection_payoff, sucker_payoff),
                "b_defects": (sucker_payoff, defection_payoff),
                "both_defect": (mutual_defection, mutual_defection),
            },
        }

    def calculate_commitment_problem(
        self,
        agent: str,
        promise: str,
    ) -> Dict[str, Any]:
        """Analyze commitment problem for an agent's promise."""
        agent_obj = self.engine.agents[agent]
        credibility = agent_obj.state.credibility
        political_capital = agent_obj.state.political_capital

        # Commitment is credible if:
        # - Agent has track record (credibility)
        # - Agent has sunk costs/audience costs (political capital at stake)
        # - Promise is self-enforcing

        credibility_factor = credibility
        audience_cost_factor = political_capital * 0.5

        overall_credibility = (credibility_factor + audience_cost_factor) / 2

        return {
            "agent": agent,
            "promise": promise,
            "credibility_score": overall_credibility,
            "is_credible": overall_credibility > 0.6,
            "factors": {
                "track_record": credibility_factor,
                "audience_costs": audience_cost_factor,
            },
            "recommendation": "Promise is likely credible" if overall_credibility > 0.6 else "Promise may not be believed",
        }


# =============================================================================
# Enhanced Simulation Engine
# =============================================================================


class EnhancedSimulationEngine(SimulationEngine):
    """Simulation engine with coalition and strategic analysis capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional managers
        self.coalition_manager = CoalitionManager(self)
        self.commitment_tracker = CommitmentTracker(self)
        self.audience_manager = AudienceCostManager(self)
        self.info_manager = InformationManager(self)
        self.game_analyzer = GameTheoreticAnalyzer(self)

    async def step(self) -> List[Action]:
        """Enhanced step with coalition and commitment tracking."""
        # Update coalition power
        self.coalition_manager.update_power_metrics()

        # Update audiences
        self.audience_manager.update_all_audiences()

        # Get actions
        actions = await super().step()

        # Track commitments from actions
        for action in actions:
            if action.action_type == ActionType.THREAT:
                self.commitment_tracker.record_commitment(
                    action.agent_name,
                    "threat",
                    action.content,
                    action.target,
                )
            elif action.action_type == ActionType.PROPOSAL:
                self.commitment_tracker.record_commitment(
                    action.agent_name,
                    "promise",
                    action.content,
                    action.target,
                )

            # Apply audience costs
            self.audience_manager.apply_action_cost(action)

        return actions

    def get_strategic_summary(self) -> Dict[str, Any]:
        """Get comprehensive strategic summary."""
        return {
            **self.get_summary(),
            "coalitions": {
                cid: {
                    "name": c.name,
                    "members": list(c.members),
                    "cohesion": c.cohesion,
                    "power": c.combined_power(self.coalition_manager.power_map),
                }
                for cid, c in self.coalition_manager.coalitions.items()
                if c.is_active
            },
            "power_balance": self.coalition_manager.get_coalition_balance(),
            "credibility_scores": {
                name: self.commitment_tracker.get_agent_credibility_score(name)
                for name in self.agents.keys()
            },
            "active_commitments": len([
                c for c in self.commitment_tracker.commitments.values()
                if c.is_active
            ]),
        }


# =============================================================================
# Factory for Enhanced Simulations
# =============================================================================


def create_enhanced_simulation(
    *args,
    **kwargs,
) -> EnhancedSimulationEngine:
    """Create an enhanced simulation with coalition dynamics."""
    return EnhancedSimulationEngine(*args, **kwargs)


def create_great_power_competition(
    persona_dir,
    inference_fn=None,
) -> EnhancedSimulationEngine:
    """Create a US-China great power competition scenario."""
    from pathlib import Path

    engine = EnhancedSimulationEngine(inference_fn=inference_fn)

    # Load key actors
    actors = [
        "xi_jinping",
        "jake_sullivan",
        "gina_raimondo",
        "sam_altman",
        "jensen_huang",
    ]

    persona_dir = Path(persona_dir)
    for actor in actors:
        paths = [
            persona_dir / f"{actor}.json",
            persona_dir / "enhanced" / f"{actor}.json",
            persona_dir / "finalized" / f"{actor}.json",
        ]
        for path in paths:
            if path.exists():
                engine.add_agent_from_persona(path)
                break

    # Initialize audiences for political actors
    engine.audience_manager.initialize_audience(
        "Xi Jinping",
        hawk_faction=0.4,
        dove_faction=0.2,
        backing_down_cost=0.4,
    )

    # Add private information
    engine.info_manager.add_private_info(
        "Sam Altman",
        "capability",
        {"topic": "AGI timeline"},
        {"months_to_agi": 18},
    )

    # Set world state
    engine.world.ai.frontier_model_capability = 80
    engine.world.ai.agi_proximity = 0.5
    engine.world.geopolitics.escalation_level = EscalationLevel.TENSION

    return engine
