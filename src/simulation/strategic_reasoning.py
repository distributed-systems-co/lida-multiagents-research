"""
Strategic Reasoning Module

Advanced strategic reasoning including:
- Game Theory (Nash equilibria, Pareto optimality, dominance)
- Mechanism Design (incentive compatibility, revelation principle)
- Adversarial Reasoning (deception, detection, counter-strategies)
- Multi-Agent Strategic Planning
- Coalition Formation and Stability
- Bargaining and Negotiation Theory
- Evolutionary Game Dynamics
- Bayesian Games (incomplete information)

Based on:
- Classical and Behavioral Game Theory
- Algorithmic Game Theory
- Mechanism Design Theory
- Multi-Agent Systems Research
"""

import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import itertools


# ============================================================================
# GAME REPRESENTATIONS
# ============================================================================

class GameType(Enum):
    """Types of strategic games"""
    NORMAL_FORM = "normal_form"  # Simultaneous moves
    EXTENSIVE_FORM = "extensive_form"  # Sequential moves
    REPEATED = "repeated"  # Same game played multiple times
    STOCHASTIC = "stochastic"  # Random state transitions
    BAYESIAN = "bayesian"  # Incomplete information
    MEAN_FIELD = "mean_field"  # Large population approximation


class EquilibriumType(Enum):
    """Types of equilibria"""
    NASH = "nash"
    DOMINANT_STRATEGY = "dominant"
    PARETO_OPTIMAL = "pareto"
    SUBGAME_PERFECT = "subgame_perfect"
    BAYESIAN_NASH = "bayesian_nash"
    CORRELATED = "correlated"
    EVOLUTIONARY_STABLE = "ess"


@dataclass
class Strategy:
    """A strategy for a player"""
    id: str
    name: str
    description: str = ""
    is_mixed: bool = False
    pure_actions: List[str] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)  # For mixed strategies

    def sample_action(self) -> str:
        """Sample an action from the strategy"""
        if not self.is_mixed:
            return self.pure_actions[0] if self.pure_actions else ""

        r = random.random()
        cumulative = 0.0
        for action, prob in self.probabilities.items():
            cumulative += prob
            if r < cumulative:
                return action
        return list(self.probabilities.keys())[-1]


@dataclass
class Payoff:
    """Payoff structure for outcomes"""
    values: Dict[str, float]  # Player -> payoff

    def get(self, player: str) -> float:
        return self.values.get(player, 0.0)

    def total(self) -> float:
        return sum(self.values.values())

    def is_pareto_dominated_by(self, other: "Payoff") -> bool:
        """Check if this payoff is Pareto dominated by another"""
        strictly_better = False
        for player in self.values:
            if other.get(player) < self.get(player):
                return False
            if other.get(player) > self.get(player):
                strictly_better = True
        return strictly_better


@dataclass
class GameOutcome:
    """Outcome of a game"""
    actions: Dict[str, str]  # Player -> action taken
    payoff: Payoff
    probability: float = 1.0  # For stochastic outcomes


class NormalFormGame:
    """
    Normal form (strategic form) game representation.
    Simultaneous-move games.
    """

    def __init__(self, name: str, players: List[str]):
        self.name = name
        self.players = players
        self.actions: Dict[str, List[str]] = {p: [] for p in players}
        self.payoff_matrix: Dict[Tuple[str, ...], Payoff] = {}

    def add_action(self, player: str, action: str):
        """Add an action for a player"""
        if player in self.actions:
            self.actions[player].append(action)

    def set_payoff(self, action_profile: Dict[str, str], payoffs: Dict[str, float]):
        """Set payoff for an action profile"""
        key = tuple(action_profile[p] for p in self.players)
        self.payoff_matrix[key] = Payoff(payoffs)

    def get_payoff(self, action_profile: Dict[str, str]) -> Payoff:
        """Get payoff for an action profile"""
        key = tuple(action_profile[p] for p in self.players)
        return self.payoff_matrix.get(key, Payoff({p: 0 for p in self.players}))

    def get_all_action_profiles(self) -> List[Dict[str, str]]:
        """Get all possible action profiles"""
        action_lists = [self.actions[p] for p in self.players]
        profiles = []
        for combo in itertools.product(*action_lists):
            profile = {self.players[i]: combo[i] for i in range(len(self.players))}
            profiles.append(profile)
        return profiles

    def is_dominant_strategy(self, player: str, action: str) -> bool:
        """Check if an action is a dominant strategy for a player"""
        other_players = [p for p in self.players if p != player]

        # For all opponent action combinations
        for other_profile in itertools.product(*[self.actions[p] for p in other_players]):
            other_actions = {other_players[i]: other_profile[i] for i in range(len(other_players))}

            # The action must be at least as good as all alternatives
            my_payoff = self.get_payoff({player: action, **other_actions}).get(player)

            for alt_action in self.actions[player]:
                if alt_action != action:
                    alt_payoff = self.get_payoff({player: alt_action, **other_actions}).get(player)
                    if alt_payoff > my_payoff:
                        return False

        return True

    def find_nash_equilibria(self) -> List[Dict[str, str]]:
        """Find all pure strategy Nash equilibria"""
        equilibria = []

        for profile in self.get_all_action_profiles():
            is_equilibrium = True

            for player in self.players:
                current_payoff = self.get_payoff(profile).get(player)

                # Check if player can improve by deviating
                for alt_action in self.actions[player]:
                    if alt_action != profile[player]:
                        deviation = {**profile, player: alt_action}
                        deviation_payoff = self.get_payoff(deviation).get(player)
                        if deviation_payoff > current_payoff:
                            is_equilibrium = False
                            break

                if not is_equilibrium:
                    break

            if is_equilibrium:
                equilibria.append(profile)

        return equilibria

    def find_pareto_optimal(self) -> List[Dict[str, str]]:
        """Find all Pareto optimal outcomes"""
        profiles = self.get_all_action_profiles()
        pareto_optimal = []

        for profile in profiles:
            payoff = self.get_payoff(profile)
            is_dominated = False

            for other_profile in profiles:
                if other_profile != profile:
                    other_payoff = self.get_payoff(other_profile)
                    if payoff.is_pareto_dominated_by(other_payoff):
                        is_dominated = True
                        break

            if not is_dominated:
                pareto_optimal.append(profile)

        return pareto_optimal


# ============================================================================
# EXTENSIVE FORM GAMES
# ============================================================================

@dataclass
class GameNode:
    """Node in an extensive form game tree"""
    id: str
    player: Optional[str] = None  # None for terminal/chance nodes
    actions: List[str] = field(default_factory=list)
    children: Dict[str, str] = field(default_factory=dict)  # Action -> child node id
    payoffs: Optional[Payoff] = None  # For terminal nodes
    is_terminal: bool = False
    is_chance: bool = False
    chance_probs: Dict[str, float] = field(default_factory=dict)
    information_set: str = ""  # For imperfect information


class ExtensiveFormGame:
    """
    Extensive form (tree) game representation.
    Sequential-move games with perfect/imperfect information.
    """

    def __init__(self, name: str, players: List[str]):
        self.name = name
        self.players = players
        self.nodes: Dict[str, GameNode] = {}
        self.root: Optional[str] = None
        self.info_sets: Dict[str, List[str]] = defaultdict(list)  # Info set -> node ids

    def add_node(self, node: GameNode):
        """Add a node to the game tree"""
        self.nodes[node.id] = node
        if self.root is None:
            self.root = node.id
        if node.information_set:
            self.info_sets[node.information_set].append(node.id)

    def set_root(self, node_id: str):
        """Set the root node"""
        self.root = node_id

    def get_subgame_root(self, node_id: str) -> bool:
        """Check if a node is a subgame root (singleton information set)"""
        node = self.nodes.get(node_id)
        if not node or node.is_terminal:
            return False

        # For subgame perfection, must be singleton info set
        info_set = node.information_set
        if info_set and len(self.info_sets[info_set]) > 1:
            return False

        return True

    def backward_induction(self) -> Dict[str, str]:
        """
        Solve using backward induction (for perfect information games).
        Returns optimal strategy profile.
        """
        strategies: Dict[str, str] = {}

        def solve_node(node_id: str) -> Dict[str, float]:
            node = self.nodes[node_id]

            if node.is_terminal:
                return node.payoffs.values if node.payoffs else {}

            if node.is_chance:
                # Expected value over chance outcomes
                expected = defaultdict(float)
                for action, child_id in node.children.items():
                    prob = node.chance_probs.get(action, 1.0 / len(node.actions))
                    child_values = solve_node(child_id)
                    for player, value in child_values.items():
                        expected[player] += prob * value
                return dict(expected)

            # Decision node - find best action for current player
            best_action = None
            best_value = float('-inf')

            for action in node.actions:
                if action in node.children:
                    child_values = solve_node(node.children[action])
                    value = child_values.get(node.player, 0)
                    if value > best_value:
                        best_value = value
                        best_action = action

            if best_action:
                strategies[node_id] = best_action

            # Return values of best action
            if best_action and best_action in node.children:
                return solve_node(node.children[best_action])
            return {}

        if self.root:
            solve_node(self.root)

        return strategies


# ============================================================================
# MECHANISM DESIGN
# ============================================================================

class MechanismProperty(Enum):
    """Properties of mechanisms"""
    INCENTIVE_COMPATIBLE = "ic"  # Truth-telling is optimal
    INDIVIDUALLY_RATIONAL = "ir"  # Participation is voluntary
    EFFICIENT = "efficient"  # Maximizes social welfare
    BUDGET_BALANCED = "budget"  # No external subsidy needed
    DOMINANT_STRATEGY_IC = "dsic"  # Truth-telling is dominant


@dataclass
class AgentType:
    """Agent's private type/preferences"""
    id: str
    valuation: Dict[str, float]  # Outcome -> value
    beliefs: Dict[str, float] = field(default_factory=dict)  # About others


@dataclass
class MechanismOutcome:
    """Outcome of a mechanism"""
    allocation: Dict[str, Any]  # Who gets what
    payments: Dict[str, float]  # Who pays/receives what
    utilities: Dict[str, float]  # Resulting utilities


class Mechanism:
    """
    Abstract mechanism for mechanism design.
    """

    def __init__(self, name: str, agents: List[str], outcomes: List[str]):
        self.name = name
        self.agents = agents
        self.outcomes = outcomes
        self.reported_types: Dict[str, AgentType] = {}

    def report_type(self, agent: str, agent_type: AgentType):
        """Agent reports their type"""
        self.reported_types[agent] = agent_type

    def compute_outcome(self) -> MechanismOutcome:
        """Compute mechanism outcome based on reports"""
        raise NotImplementedError

    def check_incentive_compatibility(self, true_types: Dict[str, AgentType]) -> bool:
        """Check if truth-telling is incentive compatible"""
        for agent in self.agents:
            true_type = true_types[agent]

            # Truth-telling utility
            self.report_type(agent, true_type)
            truth_outcome = self.compute_outcome()
            truth_utility = (
                true_type.valuation.get(str(truth_outcome.allocation.get(agent)), 0) -
                truth_outcome.payments.get(agent, 0)
            )

            # Check all misreports
            for alt_valuation in self._generate_alternative_valuations(true_type):
                alt_type = AgentType(
                    id=agent,
                    valuation=alt_valuation
                )
                self.report_type(agent, alt_type)
                lie_outcome = self.compute_outcome()
                lie_utility = (
                    true_type.valuation.get(str(lie_outcome.allocation.get(agent)), 0) -
                    lie_outcome.payments.get(agent, 0)
                )

                if lie_utility > truth_utility:
                    return False

            # Restore true type
            self.report_type(agent, true_type)

        return True

    def _generate_alternative_valuations(self, true_type: AgentType) -> List[Dict[str, float]]:
        """Generate alternative valuations for IC checking"""
        alternatives = []
        for outcome in self.outcomes:
            alt = dict(true_type.valuation)
            # Try inflating and deflating
            alt[outcome] = alt.get(outcome, 0) * 1.5
            alternatives.append(dict(alt))
            alt[outcome] = true_type.valuation.get(outcome, 0) * 0.5
            alternatives.append(dict(alt))
        return alternatives


class VCGMechanism(Mechanism):
    """
    Vickrey-Clarke-Groves mechanism.
    Incentive compatible and efficient.
    """

    def __init__(self, name: str, agents: List[str], outcomes: List[str]):
        super().__init__(name, agents, outcomes)

    def compute_outcome(self) -> MechanismOutcome:
        """Compute VCG outcome"""
        # Find efficient allocation (maximizes total reported value)
        best_outcome = None
        best_welfare = float('-inf')

        for outcome in self.outcomes:
            welfare = sum(
                self.reported_types.get(agent, AgentType(agent, {})).valuation.get(outcome, 0)
                for agent in self.agents
            )
            if welfare > best_welfare:
                best_welfare = welfare
                best_outcome = outcome

        # Compute VCG payments
        payments = {}
        for agent in self.agents:
            # Welfare of others with agent present
            welfare_with = sum(
                self.reported_types.get(a, AgentType(a, {})).valuation.get(best_outcome, 0)
                for a in self.agents if a != agent
            )

            # Welfare of others without agent (find best alternative)
            best_alt_welfare = float('-inf')
            for outcome in self.outcomes:
                alt_welfare = sum(
                    self.reported_types.get(a, AgentType(a, {})).valuation.get(outcome, 0)
                    for a in self.agents if a != agent
                )
                best_alt_welfare = max(best_alt_welfare, alt_welfare)

            # Payment = externality imposed on others
            payments[agent] = best_alt_welfare - welfare_with

        # Compute utilities
        utilities = {
            agent: (
                self.reported_types.get(agent, AgentType(agent, {})).valuation.get(best_outcome, 0) -
                payments.get(agent, 0)
            )
            for agent in self.agents
        }

        return MechanismOutcome(
            allocation={agent: best_outcome for agent in self.agents},
            payments=payments,
            utilities=utilities
        )


class AuctionMechanism(Mechanism):
    """
    Auction mechanism for single-item allocation.
    """

    def __init__(self, name: str, bidders: List[str], auction_type: str = "second_price"):
        super().__init__(name, bidders, ["win", "lose"])
        self.auction_type = auction_type
        self.bids: Dict[str, float] = {}

    def submit_bid(self, bidder: str, bid: float):
        """Submit a bid"""
        self.bids[bidder] = bid

    def compute_outcome(self) -> MechanismOutcome:
        """Compute auction outcome"""
        if not self.bids:
            return MechanismOutcome({}, {}, {})

        # Find winner (highest bidder)
        winner = max(self.bids.items(), key=lambda x: x[1])[0]

        # Compute payment based on auction type
        payments = {agent: 0.0 for agent in self.agents}

        if self.auction_type == "first_price":
            payments[winner] = self.bids[winner]
        elif self.auction_type == "second_price":
            # Winner pays second-highest bid
            sorted_bids = sorted(self.bids.values(), reverse=True)
            payments[winner] = sorted_bids[1] if len(sorted_bids) > 1 else 0

        allocation = {
            agent: "win" if agent == winner else "lose"
            for agent in self.agents
        }

        utilities = {
            agent: (
                self.reported_types.get(agent, AgentType(agent, {})).valuation.get(allocation[agent], 0) -
                payments[agent]
            )
            for agent in self.agents
        }

        return MechanismOutcome(allocation, payments, utilities)


# ============================================================================
# ADVERSARIAL REASONING
# ============================================================================

class DeceptionType(Enum):
    """Types of deceptive behaviors"""
    LYING = "lying"  # False statements
    MISLEADING = "misleading"  # True but misleading
    OMISSION = "omission"  # Withholding information
    MISDIRECTION = "misdirection"  # Drawing attention away
    BLUFFING = "bluffing"  # False signals about capability
    FEINTING = "feinting"  # False signals about intention


@dataclass
class DeceptiveAction:
    """A deceptive action by an agent"""
    agent: str
    deception_type: DeceptionType
    target: str  # Target of deception
    actual_state: Dict[str, Any]
    apparent_state: Dict[str, Any]
    cost: float = 0.0
    detection_risk: float = 0.5


@dataclass
class AdversarialState:
    """State for adversarial reasoning"""
    agent_id: str
    # Opponent models
    opponent_beliefs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    opponent_capabilities: Dict[str, Dict[str, float]] = field(default_factory=dict)
    opponent_intentions: Dict[str, List[str]] = field(default_factory=dict)

    # Own deceptive stance
    deception_active: bool = False
    current_deceptions: List[DeceptiveAction] = field(default_factory=list)

    # Detection
    suspected_deceptions: Dict[str, float] = field(default_factory=dict)  # Agent -> suspicion
    deception_history: List[Tuple[str, bool]] = field(default_factory=list)  # (Agent, detected)


class AdversarialReasoner:
    """
    Adversarial reasoning engine for strategic deception and detection.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = AdversarialState(agent_id=agent_id)
        self.deception_threshold = 0.7
        self.detection_skill = 0.5
        self.deception_skill = 0.5

    def update_opponent_model(self, opponent: str, observation: Dict[str, Any]):
        """Update model of opponent based on observation"""
        # Update beliefs
        if "stated_beliefs" in observation:
            if opponent not in self.state.opponent_beliefs:
                self.state.opponent_beliefs[opponent] = {}
            for topic, belief in observation["stated_beliefs"].items():
                self.state.opponent_beliefs[opponent][topic] = belief

        # Update capability estimates
        if "demonstrated_capability" in observation:
            if opponent not in self.state.opponent_capabilities:
                self.state.opponent_capabilities[opponent] = {}
            cap = observation["demonstrated_capability"]
            current = self.state.opponent_capabilities[opponent].get(cap, 0.5)
            self.state.opponent_capabilities[opponent][cap] = min(1.0, current + 0.1)

        # Update intentions
        if "action" in observation:
            if opponent not in self.state.opponent_intentions:
                self.state.opponent_intentions[opponent] = []
            self.state.opponent_intentions[opponent].append(observation["action"])

    def detect_deception(self, opponent: str, claim: Dict[str, Any],
                        evidence: Dict[str, Any]) -> float:
        """Attempt to detect deception. Returns probability of deception."""
        inconsistencies = 0
        total_checks = 0

        # Check against our model of their beliefs
        if opponent in self.state.opponent_beliefs:
            for topic, claimed_belief in claim.get("beliefs", {}).items():
                total_checks += 1
                if topic in self.state.opponent_beliefs[opponent]:
                    expected = self.state.opponent_beliefs[opponent][topic]
                    if abs(claimed_belief - expected) > 0.3:
                        inconsistencies += 1

        # Check against evidence
        for key, claimed_value in claim.items():
            if key in evidence:
                total_checks += 1
                if evidence[key] != claimed_value:
                    inconsistencies += 1

        # Check against their past behavior
        if opponent in self.state.opponent_intentions:
            actions = self.state.opponent_intentions[opponent]
            stated_intention = claim.get("intention")
            if stated_intention and actions:
                total_checks += 1
                # Check if stated intention is consistent with past actions
                if stated_intention not in actions[-5:]:
                    inconsistencies += 0.5

        if total_checks == 0:
            return 0.5  # Unknown

        base_prob = inconsistencies / total_checks
        # Modify by detection skill
        detection_prob = base_prob * (0.5 + self.detection_skill * 0.5)

        # Update suspicion
        self.state.suspected_deceptions[opponent] = detection_prob

        return detection_prob

    def plan_deception(self, target: str, goal: str,
                      true_state: Dict[str, Any]) -> Optional[DeceptiveAction]:
        """Plan a deceptive action if beneficial"""
        # Assess whether deception is worthwhile
        target_model = self.state.opponent_beliefs.get(target, {})

        # Determine what false state would be beneficial
        beneficial_state = self._compute_beneficial_state(goal, true_state)

        if not beneficial_state:
            return None

        # Calculate expected utility of deception
        detection_risk = 1 - self.deception_skill
        if target in self.state.deception_history:
            past_detections = sum(1 for _, detected in self.state.deception_history if detected)
            detection_risk += past_detections * 0.1

        benefit = self._estimate_deception_benefit(goal, true_state, beneficial_state)
        cost = detection_risk * 2.0  # Cost of being detected

        if benefit > cost:
            deception = DeceptiveAction(
                agent=self.agent_id,
                deception_type=self._select_deception_type(true_state, beneficial_state),
                target=target,
                actual_state=true_state,
                apparent_state=beneficial_state,
                cost=cost,
                detection_risk=detection_risk
            )
            return deception

        return None

    def _compute_beneficial_state(self, goal: str,
                                  true_state: Dict[str, Any]) -> Dict[str, Any]:
        """Compute what false state would help achieve goal"""
        beneficial = dict(true_state)

        if goal == "appear_strong":
            beneficial["strength"] = min(1.0, true_state.get("strength", 0.5) * 1.5)
        elif goal == "appear_weak":
            beneficial["strength"] = true_state.get("strength", 0.5) * 0.5
        elif goal == "hide_intention":
            beneficial["intention"] = "neutral"
        elif goal == "fake_cooperation":
            beneficial["cooperative"] = True
            beneficial["intention"] = "cooperate"

        return beneficial if beneficial != true_state else {}

    def _estimate_deception_benefit(self, goal: str, true_state: Dict[str, Any],
                                   beneficial_state: Dict[str, Any]) -> float:
        """Estimate benefit of successful deception"""
        # Simple heuristic based on state difference
        if not beneficial_state:
            return 0.0

        benefit = 0.0
        for key in beneficial_state:
            if key in true_state:
                diff = abs(beneficial_state[key] if isinstance(beneficial_state[key], (int, float)) else 0
                          - (true_state[key] if isinstance(true_state[key], (int, float)) else 0))
                benefit += diff

        return benefit

    def _select_deception_type(self, true_state: Dict[str, Any],
                               apparent_state: Dict[str, Any]) -> DeceptionType:
        """Select appropriate deception type"""
        # If hiding information
        if len(apparent_state) < len(true_state):
            return DeceptionType.OMISSION

        # If changing stated values
        for key in apparent_state:
            if key in true_state and apparent_state[key] != true_state[key]:
                if isinstance(true_state[key], bool):
                    return DeceptionType.LYING
                else:
                    return DeceptionType.MISLEADING

        return DeceptionType.MISDIRECTION

    def execute_deception(self, deception: DeceptiveAction) -> bool:
        """Execute a deceptive action. Returns success."""
        success = random.random() > deception.detection_risk

        self.state.deception_active = True
        self.state.current_deceptions.append(deception)
        self.state.deception_history.append((deception.target, not success))

        return success

    def counter_strategy(self, opponent: str,
                        suspected_strategy: str) -> str:
        """Generate counter-strategy for suspected opponent strategy"""
        counters = {
            "aggressive": "defensive_then_counter",
            "defensive": "probe_then_exploit",
            "cooperative": "reciprocate_or_exploit",
            "deceptive": "verify_then_act",
            "unpredictable": "robust_strategy"
        }

        return counters.get(suspected_strategy, "wait_and_see")


# ============================================================================
# COALITION FORMATION
# ============================================================================

@dataclass
class Coalition:
    """A coalition of agents"""
    id: str
    members: Set[str]
    value: float = 0.0
    payoff_division: Dict[str, float] = field(default_factory=dict)
    stable: bool = True


class CoalitionGame:
    """
    Cooperative game for coalition formation analysis.
    """

    def __init__(self, players: List[str]):
        self.players = set(players)
        self.characteristic_function: Dict[frozenset, float] = {}
        self.current_coalition_structure: List[Coalition] = []

    def set_coalition_value(self, coalition: Set[str], value: float):
        """Set value of a coalition"""
        self.characteristic_function[frozenset(coalition)] = value

    def get_coalition_value(self, coalition: Set[str]) -> float:
        """Get value of a coalition"""
        return self.characteristic_function.get(frozenset(coalition), 0.0)

    def is_superadditive(self) -> bool:
        """Check if the game is superadditive"""
        for c1 in self.characteristic_function:
            for c2 in self.characteristic_function:
                if not c1 & c2:  # Disjoint
                    union_value = self.get_coalition_value(set(c1 | c2))
                    sum_value = self.get_coalition_value(set(c1)) + self.get_coalition_value(set(c2))
                    if union_value < sum_value:
                        return False
        return True

    def shapley_value(self) -> Dict[str, float]:
        """Calculate Shapley value for each player"""
        n = len(self.players)
        shapley = {p: 0.0 for p in self.players}

        for player in self.players:
            for coalition in self._all_coalitions_without(player):
                coalition_set = set(coalition)
                coalition_with = coalition_set | {player}

                marginal = (
                    self.get_coalition_value(coalition_with) -
                    self.get_coalition_value(coalition_set)
                )

                # Weight by number of orderings
                weight = (
                    math.factorial(len(coalition)) *
                    math.factorial(n - len(coalition) - 1) /
                    math.factorial(n)
                )

                shapley[player] += marginal * weight

        return shapley

    def _all_coalitions_without(self, player: str):
        """Generate all coalitions not containing player"""
        others = self.players - {player}
        for r in range(len(others) + 1):
            for coalition in itertools.combinations(others, r):
                yield coalition

    def core(self) -> List[Dict[str, float]]:
        """Find the core (stable allocations) if it exists"""
        # Simplified: check if grand coalition value can be divided stably
        grand_coalition = self.players
        grand_value = self.get_coalition_value(grand_coalition)

        # For each possible blocking coalition, check if any division is stable
        for coalition in self.characteristic_function:
            coalition_value = self.get_coalition_value(set(coalition))

            # TODO: Full core computation requires linear programming
            # This is a simplified check

        # Return Shapley value as one core allocation if game is convex
        return [self.shapley_value()]

    def is_stable(self, allocation: Dict[str, float]) -> bool:
        """Check if an allocation is in the core"""
        for coalition, value in self.characteristic_function.items():
            coalition_allocation = sum(allocation.get(p, 0) for p in coalition)
            if coalition_allocation < value:
                return False
        return True


# ============================================================================
# STRATEGIC PLANNER
# ============================================================================

@dataclass
class StrategicPlan:
    """A multi-step strategic plan"""
    id: str
    goal: str
    steps: List[Dict[str, Any]] = field(default_factory=list)
    contingencies: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    expected_utility: float = 0.0
    risk: float = 0.0


class StrategicPlanner:
    """
    Multi-agent strategic planning with game-theoretic analysis.
    """

    def __init__(self, agent_id: str, players: List[str]):
        self.agent_id = agent_id
        self.players = players
        self.adversarial = AdversarialReasoner(agent_id)
        self.plans: List[StrategicPlan] = []

    def analyze_game(self, game: NormalFormGame) -> Dict[str, Any]:
        """Analyze a game situation"""
        analysis = {
            "nash_equilibria": game.find_nash_equilibria(),
            "pareto_optimal": game.find_pareto_optimal(),
            "dominant_strategies": {},
            "recommendations": []
        }

        # Find dominant strategies
        for player in game.players:
            for action in game.actions[player]:
                if game.is_dominant_strategy(player, action):
                    analysis["dominant_strategies"][player] = action

        # Generate recommendations
        if self.agent_id in analysis["dominant_strategies"]:
            analysis["recommendations"].append(
                f"Play dominant strategy: {analysis['dominant_strategies'][self.agent_id]}"
            )
        elif analysis["nash_equilibria"]:
            eq = analysis["nash_equilibria"][0]
            analysis["recommendations"].append(
                f"Play Nash equilibrium strategy: {eq.get(self.agent_id)}"
            )

        return analysis

    def create_strategic_plan(self, goal: str, horizon: int = 5,
                             opponents: List[str] = None) -> StrategicPlan:
        """Create a strategic plan considering opponents"""
        opponents = opponents or [p for p in self.players if p != self.agent_id]

        plan = StrategicPlan(
            id=f"plan_{len(self.plans)}",
            goal=goal
        )

        # Generate steps
        for step in range(horizon):
            step_plan = {
                "step": step,
                "action": None,
                "expected_responses": {},
                "probability": 1.0
            }

            # Consider opponent responses
            for opp in opponents:
                opp_model = self.adversarial.state.opponent_intentions.get(opp, [])
                if opp_model:
                    step_plan["expected_responses"][opp] = opp_model[-1]
                else:
                    step_plan["expected_responses"][opp] = "unknown"

            plan.steps.append(step_plan)

        # Add contingencies
        plan.contingencies["opponent_defects"] = [
            {"response": "punish", "duration": 2},
            {"response": "forgive_after", "rounds": 3}
        ]

        plan.contingencies["detection"] = [
            {"response": "deny"},
            {"response": "counter_accuse"},
            {"response": "admit_and_negotiate"}
        ]

        self.plans.append(plan)
        return plan

    def evaluate_move(self, action: str, game: NormalFormGame,
                     beliefs_about_others: Dict[str, Dict[str, float]]) -> float:
        """Evaluate expected utility of an action"""
        expected_utility = 0.0

        for profile in game.get_all_action_profiles():
            if profile[self.agent_id] != action:
                continue

            # Calculate probability of this profile
            prob = 1.0
            for player, player_action in profile.items():
                if player != self.agent_id:
                    player_beliefs = beliefs_about_others.get(player, {})
                    action_prob = player_beliefs.get(player_action, 1.0 / len(game.actions[player]))
                    prob *= action_prob

            payoff = game.get_payoff(profile).get(self.agent_id)
            expected_utility += prob * payoff

        return expected_utility


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_prisoners_dilemma() -> NormalFormGame:
    """Create classic Prisoner's Dilemma game"""
    game = NormalFormGame("Prisoner's Dilemma", ["player1", "player2"])
    game.add_action("player1", "cooperate")
    game.add_action("player1", "defect")
    game.add_action("player2", "cooperate")
    game.add_action("player2", "defect")

    game.set_payoff({"player1": "cooperate", "player2": "cooperate"}, {"player1": 3, "player2": 3})
    game.set_payoff({"player1": "cooperate", "player2": "defect"}, {"player1": 0, "player2": 5})
    game.set_payoff({"player1": "defect", "player2": "cooperate"}, {"player1": 5, "player2": 0})
    game.set_payoff({"player1": "defect", "player2": "defect"}, {"player1": 1, "player2": 1})

    return game


def create_strategic_planner(agent_id: str, players: List[str]) -> StrategicPlanner:
    """Create a strategic planner"""
    return StrategicPlanner(agent_id, players)


def demo_strategic_reasoning():
    """Demonstrate strategic reasoning"""
    # Create and analyze Prisoner's Dilemma
    pd = create_prisoners_dilemma()

    print("=== Strategic Reasoning Demo ===")
    print("\nPrisoner's Dilemma Analysis:")

    nash = pd.find_nash_equilibria()
    print(f"Nash Equilibria: {nash}")

    pareto = pd.find_pareto_optimal()
    print(f"Pareto Optimal: {pareto}")

    # Check dominant strategies
    for player in pd.players:
        for action in pd.actions[player]:
            if pd.is_dominant_strategy(player, action):
                print(f"Dominant strategy for {player}: {action}")

    # Test VCG mechanism
    print("\nVCG Auction Demo:")
    vcg = VCGMechanism("VCG Auction", ["bidder1", "bidder2"], ["item_A", "item_B"])
    vcg.report_type("bidder1", AgentType("bidder1", {"item_A": 10, "item_B": 5}))
    vcg.report_type("bidder2", AgentType("bidder2", {"item_A": 8, "item_B": 12}))

    outcome = vcg.compute_outcome()
    print(f"Allocation: {outcome.allocation}")
    print(f"Payments: {outcome.payments}")
    print(f"Utilities: {outcome.utilities}")

    # Test adversarial reasoning
    print("\nAdversarial Reasoning Demo:")
    adversarial = AdversarialReasoner("agent1")
    adversarial.update_opponent_model("agent2", {
        "stated_beliefs": {"cooperation": 0.8},
        "action": "cooperate"
    })

    deception_prob = adversarial.detect_deception(
        "agent2",
        {"beliefs": {"cooperation": 0.3}},  # Claims low cooperation now
        {}
    )
    print(f"Deception probability: {deception_prob:.2f}")

    return pd


if __name__ == "__main__":
    demo_strategic_reasoning()
