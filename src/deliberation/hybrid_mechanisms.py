"""
Novel Hybrid Deliberation Mechanisms

Combines multiple mechanism design principles:
- Futarchy: "Vote on values, bet on beliefs"
- Schelling Points: Coordination without communication
- Information Elicitation: Proper scoring rules for honest revelation
- Liquid Democracy: Delegatable votes with expertise weighting
- Adversarial Collaboration: Structured disagreement protocols
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import logging

from .mechanisms import Position, Belief, AgentState, DeliberationResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# FUTARCHY: Vote on Values, Bet on Beliefs
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionalMarket:
    """Market for P(outcome|decision)."""
    decision: str
    outcome_metric: str
    price: float = 0.5
    liquidity: float = 100.0
    quantity: float = 0.0


class Futarchy:
    """
    Futarchy mechanism: Vote on values, bet on beliefs.

    1. Collective votes on what metric matters (the "value")
    2. Prediction markets estimate metric under each decision
    3. Decision with highest predicted metric value wins

    Example:
    - Value: "AI progress that benefits humanity"
    - Decisions: [Safety First, Capability First, Balanced]
    - Markets predict: P(beneficial outcome | each decision)
    - Choose decision with highest predicted beneficial outcome
    """

    def __init__(
        self,
        decisions: List[Position],
        outcome_metric: str = "success_probability",
        liquidity: float = 100.0
    ):
        self.decisions = {d.id: d for d in decisions}
        self.outcome_metric = outcome_metric
        self.liquidity = liquidity

        # Conditional markets: one per decision
        self.markets: Dict[str, ConditionalMarket] = {
            d.id: ConditionalMarket(
                decision=d.id,
                outcome_metric=outcome_metric,
                liquidity=liquidity
            )
            for d in decisions
        }

        self.agents: Dict[str, AgentState] = {}
        self.value_weights: Dict[str, float] = {}  # Agent weights on the outcome metric

    def register_agent(self, agent_id: str, budget: float = 100.0):
        self.agents[agent_id] = AgentState(agent_id=agent_id, budget=budget)

    def set_value_weight(self, agent_id: str, weight: float):
        """How much this agent cares about the outcome metric (0-1)."""
        self.value_weights[agent_id] = max(0, min(1, weight))

    def _market_price(self, market: ConditionalMarket) -> float:
        """LMSR price for conditional market."""
        b = market.liquidity
        return 1 / (1 + math.exp(-market.quantity / b))

    def trade_conditional(
        self,
        agent_id: str,
        decision_id: str,
        shares: float  # Positive = bet outcome happens if decision made
    ) -> Tuple[bool, float]:
        """Trade in conditional market."""
        if agent_id not in self.agents:
            return False, 0.0
        if decision_id not in self.markets:
            return False, 0.0

        agent = self.agents[agent_id]
        market = self.markets[decision_id]

        # Calculate cost using LMSR
        b = market.liquidity
        old_cost = b * math.log(1 + math.exp(market.quantity / b))
        new_quantity = market.quantity + shares
        new_cost = b * math.log(1 + math.exp(new_quantity / b))
        cost = new_cost - old_cost

        if cost > agent.budget:
            return False, 0.0

        # Execute trade
        agent.budget -= cost
        market.quantity = new_quantity
        market.price = self._market_price(market)

        return True, cost

    def get_decision_scores(self) -> Dict[str, float]:
        """Get predicted outcome probability for each decision."""
        return {did: self._market_price(m) for did, m in self.markets.items()}

    def get_results(self) -> DeliberationResult:
        scores = self.get_decision_scores()

        # Weight by value weights if available
        if self.value_weights:
            total_weight = sum(self.value_weights.values())
            if total_weight > 0:
                # Aggregate value weights (how much agents care about metric)
                avg_weight = total_weight / len(self.value_weights)
                # Scale scores by collective value weight
                scores = {k: v * avg_weight for k, v in scores.items()}

        winner_id = max(scores, key=scores.get) if scores else None

        # Consensus from price spread
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            spread = max_score - min_score
            consensus = spread  # Higher spread = clearer winner
        else:
            consensus = 0.0

        return DeliberationResult(
            winning_position=self.decisions.get(winner_id),
            position_scores=scores,
            agent_contributions={
                aid: {"budget_remaining": a.budget, "value_weight": self.value_weights.get(aid, 0.5)}
                for aid, a in self.agents.items()
            },
            consensus_strength=consensus,
            rounds_completed=1,
            mechanism="futarchy",
            metadata={"outcome_metric": self.outcome_metric}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SCHELLING POINT DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

class SchellingPointMechanism:
    """
    Discover natural coordination points through multiple rounds.

    Agents try to guess what others will choose. Rewards for matching
    the modal response, which surfaces "obvious" consensus positions.

    Inspired by Schelling's focal point theory.
    """

    def __init__(self, positions: List[Position], num_rounds: int = 3):
        self.positions = {p.id: p for p in positions}
        self.num_rounds = num_rounds
        self.agents: Dict[str, AgentState] = {}
        self.round_choices: List[Dict[str, str]] = []  # round -> agent -> choice
        self.coordination_scores: Dict[str, float] = {}  # agent -> total coordination score

    def register_agent(self, agent_id: str):
        self.agents[agent_id] = AgentState(agent_id=agent_id)
        self.coordination_scores[agent_id] = 0.0

    def submit_choice(
        self,
        round_num: int,
        agent_id: str,
        position_id: str
    ) -> bool:
        """Submit coordination attempt."""
        if agent_id not in self.agents:
            return False
        if position_id not in self.positions:
            return False

        while len(self.round_choices) <= round_num:
            self.round_choices.append({})

        self.round_choices[round_num][agent_id] = position_id
        return True

    def score_round(self, round_num: int) -> Dict[str, int]:
        """Score a round - points for matching the mode."""
        if round_num >= len(self.round_choices):
            return {}

        choices = self.round_choices[round_num]
        if not choices:
            return {}

        # Find mode (most common choice)
        from collections import Counter
        counts = Counter(choices.values())
        mode_choice, mode_count = counts.most_common(1)[0]

        # Score agents
        scores = {}
        for aid, choice in choices.items():
            if choice == mode_choice:
                # Coordination bonus scaled by how dominant the mode was
                scores[aid] = mode_count / len(choices)
            else:
                scores[aid] = 0

            self.coordination_scores[aid] += scores[aid]

        return scores

    def get_focal_point(self) -> Optional[str]:
        """Get the emergent focal point across all rounds."""
        all_choices = []
        for round_choices in self.round_choices:
            all_choices.extend(round_choices.values())

        if not all_choices:
            return None

        from collections import Counter
        counts = Counter(all_choices)
        return counts.most_common(1)[0][0]

    def get_results(self) -> DeliberationResult:
        focal_point_id = self.get_focal_point()

        # Position scores based on selection frequency
        all_choices = []
        for rc in self.round_choices:
            all_choices.extend(rc.values())

        from collections import Counter
        counts = Counter(all_choices)
        total = len(all_choices) or 1
        scores = {pid: counts.get(pid, 0) / total for pid in self.positions}

        # Consensus from coordination scores
        if self.coordination_scores:
            avg_coord = sum(self.coordination_scores.values()) / len(self.coordination_scores)
            consensus = avg_coord / self.num_rounds  # Normalize by rounds
        else:
            consensus = 0.0

        return DeliberationResult(
            winning_position=self.positions.get(focal_point_id),
            position_scores=scores,
            agent_contributions={
                aid: {"coordination_score": score, "final_choice": self.round_choices[-1].get(aid) if self.round_choices else None}
                for aid, score in self.coordination_scores.items()
            },
            consensus_strength=consensus,
            rounds_completed=len(self.round_choices),
            mechanism="schelling_point",
            metadata={"focal_point": focal_point_id}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PEER PREDICTION (Information Elicitation)
# ═══════════════════════════════════════════════════════════════════════════════

class PeerPrediction:
    """
    Peer Prediction mechanism for eliciting honest beliefs.

    Agents report their belief AND predict what others will report.
    Rewards based on how well predictions match actual peer reports.

    This incentivizes truthful revelation even without verifiable ground truth.
    Based on Miller, Resnick, Zeckhauser (2005).
    """

    def __init__(self, positions: List[Position]):
        self.positions = {p.id: p for p in positions}
        self.agents: Dict[str, AgentState] = {}

        # Reports: agent -> position_id (their belief)
        self.own_reports: Dict[str, str] = {}

        # Predictions: agent -> {position_id: probability others choose it}
        self.peer_predictions: Dict[str, Dict[str, float]] = {}

        self.scores: Dict[str, float] = {}

    def register_agent(self, agent_id: str):
        self.agents[agent_id] = AgentState(agent_id=agent_id)
        self.scores[agent_id] = 0.0

    def submit_report(
        self,
        agent_id: str,
        own_belief: str,  # Position they believe in
        peer_prediction: Dict[str, float]  # P(others choose each position)
    ):
        """Submit belief and prediction about peers."""
        if agent_id not in self.agents:
            return False

        self.own_reports[agent_id] = own_belief

        # Normalize predictions
        total = sum(peer_prediction.values()) or 1
        self.peer_predictions[agent_id] = {
            k: v/total for k, v in peer_prediction.items()
        }
        return True

    def compute_scores(self):
        """Compute peer prediction scores."""
        if len(self.own_reports) < 2:
            return

        # Actual distribution of peer reports
        from collections import Counter
        report_counts = Counter(self.own_reports.values())
        total_reports = len(self.own_reports)

        for agent_id in self.agents:
            if agent_id not in self.peer_predictions:
                continue

            # Score = log scoring rule on peer prediction accuracy
            # Exclude own report when computing "peer" distribution
            other_reports = [r for aid, r in self.own_reports.items() if aid != agent_id]
            if not other_reports:
                continue

            other_counts = Counter(other_reports)
            other_total = len(other_reports)

            # Compute Brier-like score (lower is better, we'll negate)
            brier = 0.0
            for pos_id in self.positions:
                predicted_prob = self.peer_predictions[agent_id].get(pos_id, 0)
                actual_prob = other_counts.get(pos_id, 0) / other_total
                brier += (predicted_prob - actual_prob) ** 2

            # Convert to score (higher is better)
            self.scores[agent_id] = 1 - (brier / len(self.positions))

    def get_results(self) -> DeliberationResult:
        self.compute_scores()

        # Aggregate beliefs weighted by prediction accuracy
        weighted_votes: Dict[str, float] = {pid: 0.0 for pid in self.positions}
        total_weight = 0.0

        for agent_id, report in self.own_reports.items():
            weight = max(0.1, self.scores.get(agent_id, 0.5))  # Min weight to avoid zeros
            weighted_votes[report] = weighted_votes.get(report, 0) + weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            scores = {k: v/total_weight for k, v in weighted_votes.items()}
        else:
            scores = {k: 1/len(self.positions) for k in self.positions}

        winner_id = max(scores, key=scores.get)

        # Consensus from score variance
        score_vals = list(self.scores.values())
        if score_vals:
            mean_score = sum(score_vals) / len(score_vals)
            consensus = mean_score  # Higher avg score = better calibration overall
        else:
            consensus = 0.0

        return DeliberationResult(
            winning_position=self.positions.get(winner_id),
            position_scores=scores,
            agent_contributions={
                aid: {
                    "report": self.own_reports.get(aid),
                    "prediction_score": self.scores.get(aid, 0),
                    "peer_predictions": self.peer_predictions.get(aid, {})
                }
                for aid in self.agents
            },
            consensus_strength=consensus,
            rounds_completed=1,
            mechanism="peer_prediction",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# LIQUID DEMOCRACY WITH EXPERTISE DOMAINS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Delegation:
    """A vote delegation."""
    from_agent: str
    to_agent: str
    domain: str  # Topic/expertise area
    weight: float = 1.0


class LiquidDemocracy:
    """
    Liquid Democracy with domain-specific delegation.

    Agents can:
    1. Vote directly
    2. Delegate to trusted experts by domain
    3. Receive delegated voting power

    Voting power flows through delegation chains.
    Cycles are detected and broken.
    """

    def __init__(self, positions: List[Position], domains: List[str]):
        self.positions = {p.id: p for p in positions}
        self.domains = set(domains)
        self.agents: Dict[str, AgentState] = {}

        # Delegations by domain
        self.delegations: Dict[str, List[Delegation]] = {d: [] for d in domains}

        # Direct votes: agent -> position -> weight
        self.direct_votes: Dict[str, Dict[str, float]] = {}

        # Expertise scores: agent -> domain -> score
        self.expertise: Dict[str, Dict[str, float]] = {}

    def register_agent(
        self,
        agent_id: str,
        expertise: Optional[Dict[str, float]] = None
    ):
        self.agents[agent_id] = AgentState(agent_id=agent_id, budget=1.0)
        self.expertise[agent_id] = expertise or {}
        self.direct_votes[agent_id] = {}

    def delegate(
        self,
        from_agent: str,
        to_agent: str,
        domain: str,
        weight: float = 1.0
    ) -> bool:
        """Delegate voting power in a domain."""
        if from_agent not in self.agents or to_agent not in self.agents:
            return False
        if domain not in self.domains:
            return False
        if from_agent == to_agent:
            return False

        # Check for cycles
        if self._creates_cycle(from_agent, to_agent, domain):
            return False

        self.delegations[domain].append(Delegation(
            from_agent=from_agent,
            to_agent=to_agent,
            domain=domain,
            weight=weight
        ))
        return True

    def _creates_cycle(self, from_agent: str, to_agent: str, domain: str) -> bool:
        """Check if delegation would create a cycle."""
        visited = {from_agent}
        queue = [to_agent]

        while queue:
            current = queue.pop(0)
            if current in visited:
                return True
            visited.add(current)

            # Find who current delegates to
            for d in self.delegations[domain]:
                if d.from_agent == current:
                    queue.append(d.to_agent)

        return False

    def vote(
        self,
        agent_id: str,
        position_id: str,
        weight: float = 1.0
    ) -> bool:
        """Cast a direct vote."""
        if agent_id not in self.agents:
            return False
        if position_id not in self.positions:
            return False

        self.direct_votes[agent_id][position_id] = weight
        return True

    def compute_voting_power(self, domain: str) -> Dict[str, float]:
        """Compute effective voting power for each agent in domain."""
        # Start with base power (1.0 each)
        power = {aid: 1.0 for aid in self.agents}

        # Apply expertise multiplier
        for aid in self.agents:
            domain_expertise = self.expertise.get(aid, {}).get(domain, 0.5)
            power[aid] *= (0.5 + domain_expertise)  # Range: 0.5x to 1.5x

        # Flow power through delegations
        # Iterate until convergence
        for _ in range(10):  # Max iterations
            new_power = {aid: 1.0 for aid in self.agents}

            for aid in self.agents:
                domain_expertise = self.expertise.get(aid, {}).get(domain, 0.5)
                new_power[aid] *= (0.5 + domain_expertise)

            # Add delegated power
            for d in self.delegations[domain]:
                # Power flows from delegator to delegate
                delegated = power[d.from_agent] * d.weight * 0.9  # 10% friction
                new_power[d.to_agent] += delegated
                new_power[d.from_agent] -= power[d.from_agent] * d.weight

            # Check convergence
            max_diff = max(abs(new_power[a] - power[a]) for a in self.agents)
            power = new_power
            if max_diff < 0.01:
                break

        return power

    def get_results(self, primary_domain: str) -> DeliberationResult:
        """Compute results using primary domain for power calculation."""
        voting_power = self.compute_voting_power(primary_domain)

        # Tally weighted votes
        tallies: Dict[str, float] = {pid: 0.0 for pid in self.positions}

        for aid, votes in self.direct_votes.items():
            agent_power = voting_power.get(aid, 1.0)
            for pos_id, weight in votes.items():
                tallies[pos_id] += agent_power * weight

        # Normalize
        total = sum(tallies.values()) or 1
        scores = {k: v/total for k, v in tallies.items()}

        winner_id = max(scores, key=scores.get) if scores else None

        # Consensus from power concentration
        power_vals = list(voting_power.values())
        if power_vals:
            max_power = max(power_vals)
            total_power = sum(power_vals)
            concentration = max_power / total_power if total_power > 0 else 0
            consensus = 1 - concentration  # Lower concentration = more distributed = higher consensus
        else:
            consensus = 0.5

        return DeliberationResult(
            winning_position=self.positions.get(winner_id),
            position_scores=scores,
            agent_contributions={
                aid: {
                    "voting_power": voting_power.get(aid, 0),
                    "direct_votes": self.direct_votes.get(aid, {}),
                    "expertise": self.expertise.get(aid, {}),
                    "delegations_given": len([d for d in self.delegations[primary_domain] if d.from_agent == aid]),
                    "delegations_received": len([d for d in self.delegations[primary_domain] if d.to_agent == aid]),
                }
                for aid in self.agents
            },
            consensus_strength=consensus,
            rounds_completed=1,
            mechanism="liquid_democracy",
            metadata={
                "domain": primary_domain,
                "total_delegations": sum(len(d) for d in self.delegations.values())
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ADVERSARIAL COLLABORATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CritiqueExchange:
    """A structured critique between agents."""
    critic_id: str
    target_id: str
    position_id: str
    critique_strength: float  # 0-1, how damaging
    target_response_strength: float  # 0-1, how well defended
    resolved: bool = False


class AdversarialCollaboration:
    """
    Adversarial Collaboration mechanism.

    Inspired by Kahneman's adversarial collaboration protocol:
    1. Agents with opposing views pair up
    2. Each must steelman the opponent's position
    3. Each must identify cruxes (key disagreements)
    4. Critiques and defenses are evaluated
    5. Positions that survive scrutiny score higher

    This surfaces robust positions and eliminates weak ones.
    """

    def __init__(self, positions: List[Position]):
        self.positions = {p.id: p for p in positions}
        self.agents: Dict[str, AgentState] = {}

        # Agent positions: agent -> position they advocate
        self.advocated_positions: Dict[str, str] = {}

        # Steelman scores: agent -> how well they steelmanned opponent
        self.steelman_scores: Dict[str, float] = {}

        # Critique exchanges
        self.exchanges: List[CritiqueExchange] = []

        # Position robustness: accumulated from surviving critiques
        self.robustness: Dict[str, float] = {p.id: 1.0 for p in positions}

    def register_agent(self, agent_id: str, position_id: str):
        """Register agent as advocate for a position."""
        if position_id not in self.positions:
            return False
        self.agents[agent_id] = AgentState(agent_id=agent_id)
        self.advocated_positions[agent_id] = position_id
        self.steelman_scores[agent_id] = 0.0
        return True

    def record_steelman(
        self,
        agent_id: str,
        opponent_position_id: str,
        quality: float  # 0-1, how well they represented opponent
    ):
        """Record steelman attempt quality."""
        if agent_id not in self.agents:
            return
        self.steelman_scores[agent_id] = quality

    def submit_critique(
        self,
        critic_id: str,
        target_id: str,
        strength: float  # 0-1, critique strength
    ) -> bool:
        """Submit critique of another agent's position."""
        if critic_id not in self.agents or target_id not in self.agents:
            return False

        target_position = self.advocated_positions.get(target_id)
        if not target_position:
            return False

        self.exchanges.append(CritiqueExchange(
            critic_id=critic_id,
            target_id=target_id,
            position_id=target_position,
            critique_strength=strength,
            target_response_strength=0.0
        ))
        return True

    def submit_defense(
        self,
        exchange_idx: int,
        defense_strength: float
    ):
        """Respond to a critique."""
        if exchange_idx >= len(self.exchanges):
            return

        exchange = self.exchanges[exchange_idx]
        exchange.target_response_strength = defense_strength
        exchange.resolved = True

        # Update position robustness based on exchange outcome
        # If defense > critique, position gains robustness
        # If critique > defense, position loses robustness
        delta = (defense_strength - exchange.critique_strength) * 0.2
        self.robustness[exchange.position_id] = max(0.1,
            self.robustness[exchange.position_id] + delta
        )

    def get_results(self) -> DeliberationResult:
        # Final scores combine:
        # 1. Robustness from surviving critiques
        # 2. Advocate steelman quality (good faith participation)

        position_scores: Dict[str, float] = {}

        for pos_id in self.positions:
            base_robustness = self.robustness[pos_id]

            # Average steelman score of position's advocates
            advocates = [aid for aid, pid in self.advocated_positions.items() if pid == pos_id]
            if advocates:
                avg_steelman = sum(self.steelman_scores.get(a, 0) for a in advocates) / len(advocates)
            else:
                avg_steelman = 0.5

            # Combined score
            position_scores[pos_id] = base_robustness * (0.5 + avg_steelman * 0.5)

        # Normalize
        total = sum(position_scores.values()) or 1
        position_scores = {k: v/total for k, v in position_scores.items()}

        winner_id = max(position_scores, key=position_scores.get)

        # Consensus from exchange outcomes
        if self.exchanges:
            resolved = [e for e in self.exchanges if e.resolved]
            if resolved:
                # Higher consensus if critiques were well-defended
                avg_defense = sum(e.target_response_strength for e in resolved) / len(resolved)
                consensus = avg_defense
            else:
                consensus = 0.5
        else:
            consensus = 0.5

        return DeliberationResult(
            winning_position=self.positions.get(winner_id),
            position_scores=position_scores,
            agent_contributions={
                aid: {
                    "position": self.advocated_positions.get(aid),
                    "steelman_score": self.steelman_scores.get(aid, 0),
                    "critiques_given": len([e for e in self.exchanges if e.critic_id == aid]),
                    "critiques_received": len([e for e in self.exchanges if e.target_id == aid]),
                }
                for aid in self.agents
            },
            consensus_strength=consensus,
            rounds_completed=len(self.exchanges),
            mechanism="adversarial_collaboration",
            metadata={
                "robustness_scores": dict(self.robustness),
                "total_exchanges": len(self.exchanges)
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EPISTEMIC AUCTION: Combined Market + Conviction + Reputation
# ═══════════════════════════════════════════════════════════════════════════════

class EpistemicAuction:
    """
    Novel hybrid: Epistemic Auction.

    Combines:
    - Auction dynamics (bid for influence)
    - Prediction markets (prices reflect beliefs)
    - Reputation staking (accuracy affects future weight)
    - Information revelation (bids reveal private info)

    Agents bid for "epistemic influence" - the right to have their
    belief weighted in the final aggregate. Higher bids cost more
    but grant more influence. Winning bidders stake reputation on
    their positions.
    """

    def __init__(
        self,
        positions: List[Position],
        num_influence_slots: int = 5
    ):
        self.positions = {p.id: p for p in positions}
        self.num_slots = num_influence_slots
        self.agents: Dict[str, AgentState] = {}

        # Bids: agent -> (position, bid_amount, confidence)
        self.bids: Dict[str, Tuple[str, float, float]] = {}

        # Winning bids
        self.winners: List[Tuple[str, str, float, float]] = []  # (agent, position, bid, confidence)

        # Historical accuracy for reputation
        self.accuracy_history: Dict[str, List[float]] = {}

    def register_agent(
        self,
        agent_id: str,
        budget: float = 100.0,
        prior_accuracy: Optional[List[float]] = None
    ):
        self.agents[agent_id] = AgentState(agent_id=agent_id, budget=budget)
        self.accuracy_history[agent_id] = prior_accuracy or [0.5]

    def _get_reputation(self, agent_id: str) -> float:
        """Get agent's reputation from accuracy history."""
        history = self.accuracy_history.get(agent_id, [0.5])
        if not history:
            return 0.5
        # Weighted average, more recent counts more
        weights = [1.5 ** i for i in range(len(history))]
        total_weight = sum(weights)
        return sum(h * w for h, w in zip(history, weights)) / total_weight

    def submit_bid(
        self,
        agent_id: str,
        position_id: str,
        bid_amount: float,
        confidence: float  # 0-1
    ) -> Tuple[bool, str]:
        """Submit bid for epistemic influence."""
        if agent_id not in self.agents:
            return False, "Agent not registered"
        if position_id not in self.positions:
            return False, "Position not found"

        agent = self.agents[agent_id]
        if bid_amount > agent.budget:
            return False, f"Insufficient budget ({agent.budget:.2f})"

        self.bids[agent_id] = (position_id, bid_amount, confidence)
        return True, f"Bid {bid_amount:.2f} on {position_id}"

    def run_auction(self):
        """Run the auction to determine winners."""
        if not self.bids:
            return

        # Score bids: bid_amount * reputation * confidence
        scored_bids = []
        for aid, (pos, bid, conf) in self.bids.items():
            rep = self._get_reputation(aid)
            score = bid * rep * conf
            scored_bids.append((score, aid, pos, bid, conf))

        # Sort by score descending
        scored_bids.sort(reverse=True)

        # Take top N
        self.winners = []
        for score, aid, pos, bid, conf in scored_bids[:self.num_slots]:
            self.winners.append((aid, pos, bid, conf))
            # Deduct bid from budget
            self.agents[aid].budget -= bid

    def get_results(self) -> DeliberationResult:
        if not self.winners:
            self.run_auction()

        # Aggregate beliefs from winners, weighted by bid and confidence
        weighted_beliefs: Dict[str, float] = {pid: 0.0 for pid in self.positions}
        total_weight = 0.0

        for aid, pos, bid, conf in self.winners:
            rep = self._get_reputation(aid)
            weight = bid * conf * rep
            weighted_beliefs[pos] += weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            scores = {k: v/total_weight for k, v in weighted_beliefs.items()}
        else:
            scores = {k: 1/len(self.positions) for k in self.positions}

        winner_id = max(scores, key=scores.get)

        # Consensus from bid concentration
        if self.winners:
            winner_positions = [pos for _, pos, _, _ in self.winners]
            from collections import Counter
            pos_counts = Counter(winner_positions)
            max_count = max(pos_counts.values())
            consensus = max_count / len(self.winners)
        else:
            consensus = 0.0

        return DeliberationResult(
            winning_position=self.positions.get(winner_id),
            position_scores=scores,
            agent_contributions={
                aid: {
                    "bid": self.bids.get(aid, (None, 0, 0)),
                    "reputation": self._get_reputation(aid),
                    "won_slot": aid in [w[0] for w in self.winners],
                    "remaining_budget": self.agents[aid].budget
                }
                for aid in self.agents
            },
            consensus_strength=consensus,
            rounds_completed=1,
            mechanism="epistemic_auction",
            metadata={
                "num_slots": self.num_slots,
                "winners": [(w[0], w[1], w[2]) for w in self.winners]
            }
        )

    def resolve_and_update_reputation(self, actual_outcome: str):
        """After ground truth known, update reputations."""
        for aid, pos, bid, conf in self.winners:
            if pos == actual_outcome:
                accuracy = conf  # Rewarded proportional to confidence
            else:
                accuracy = 1 - conf  # Penalized proportional to confidence

            self.accuracy_history[aid].append(accuracy)
            # Keep limited history
            if len(self.accuracy_history[aid]) > 10:
                self.accuracy_history[aid] = self.accuracy_history[aid][-10:]
