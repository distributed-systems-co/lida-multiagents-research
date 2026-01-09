"""
Deliberation Mechanisms for Multi-Agent Systems

Implements auction-based and market-based consensus mechanisms:
- Quadratic Voting: Cost increases quadratically with conviction
- Prediction Markets: Agents trade on outcome probabilities
- Bayesian Aggregation: Proper scoring rule-based belief combination
- Conviction Staking: Agents stake reputation on positions
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """A position in a deliberation."""
    id: str
    name: str
    description: str

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Belief:
    """An agent's belief about a position."""
    position: Position
    probability: float  # 0-1, agent's credence
    confidence: float   # 0-1, how certain they are of their probability estimate
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)

    def expected_value(self) -> float:
        """Confidence-weighted probability."""
        return self.probability * self.confidence


@dataclass
class AgentState:
    """State of an agent in the deliberation."""
    agent_id: str
    budget: float = 100.0  # Voting/betting credits
    reputation: float = 1.0  # Track record multiplier
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)
    votes_cast: Dict[str, float] = field(default_factory=dict)

    def update_reputation(self, accuracy: float):
        """Update reputation based on prediction accuracy."""
        # Exponential moving average
        self.reputation = 0.8 * self.reputation + 0.2 * accuracy


@dataclass
class MarketState:
    """State of a prediction market."""
    prices: Dict[str, float]  # position_id -> price (0-1)
    liquidity: Dict[str, float]  # position_id -> liquidity depth
    volume: float = 0.0
    trades: List[Dict] = field(default_factory=list)


@dataclass
class DeliberationResult:
    """Result of a deliberation round."""
    winning_position: Optional[Position]
    position_scores: Dict[str, float]
    agent_contributions: Dict[str, Dict]
    consensus_strength: float  # 0-1, how strong the agreement is
    rounds_completed: int
    mechanism: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# QUADRATIC VOTING
# ═══════════════════════════════════════════════════════════════════════════════

class QuadraticVoting:
    """
    Quadratic Voting mechanism where cost = votes^2.

    Agents allocate limited voice credits across positions.
    Buying N votes for a position costs N^2 credits.
    This prevents plutocracy while allowing intensity expression.
    """

    def __init__(self, positions: List[Position], voice_credits: float = 100.0):
        self.positions = {p.id: p for p in positions}
        self.default_credits = voice_credits
        self.agents: Dict[str, AgentState] = {}
        self.votes: Dict[str, float] = {p.id: 0.0 for p in positions}

    def register_agent(self, agent_id: str, credits: Optional[float] = None):
        """Register an agent with voice credits."""
        self.agents[agent_id] = AgentState(
            agent_id=agent_id,
            budget=credits or self.default_credits
        )

    def vote_cost(self, num_votes: float) -> float:
        """Cost of casting num_votes votes (quadratic)."""
        return num_votes ** 2

    def max_votes_for_budget(self, budget: float) -> float:
        """Maximum votes purchasable with budget."""
        return math.sqrt(budget)

    def cast_votes(
        self,
        agent_id: str,
        position_id: str,
        num_votes: float
    ) -> Tuple[bool, str]:
        """
        Cast votes for a position.

        Args:
            agent_id: The voting agent
            position_id: Position to vote for
            num_votes: Number of votes (can be negative for against)

        Returns:
            (success, message)
        """
        if agent_id not in self.agents:
            return False, f"Agent {agent_id} not registered"

        if position_id not in self.positions:
            return False, f"Position {position_id} not found"

        agent = self.agents[agent_id]

        # Calculate cost (absolute value for negative votes)
        cost = self.vote_cost(abs(num_votes))

        # Check if agent already voted on this position
        existing_votes = agent.votes_cast.get(position_id, 0)
        existing_cost = self.vote_cost(abs(existing_votes))

        # New total votes for this position
        new_total = existing_votes + num_votes
        new_cost = self.vote_cost(abs(new_total))

        # Marginal cost is difference
        marginal_cost = new_cost - existing_cost

        if marginal_cost > agent.budget:
            max_additional = self.max_votes_for_budget(agent.budget + existing_cost) - abs(existing_votes)
            return False, f"Insufficient credits. Can add max {max_additional:.2f} votes"

        # Execute vote
        agent.budget -= marginal_cost
        agent.votes_cast[position_id] = new_total
        self.votes[position_id] += num_votes

        return True, f"Cast {num_votes:.2f} votes. Cost: {marginal_cost:.2f}. Remaining: {agent.budget:.2f}"

    def get_results(self) -> DeliberationResult:
        """Calculate final results."""
        # Normalize votes to scores
        total_abs_votes = sum(abs(v) for v in self.votes.values())

        if total_abs_votes == 0:
            scores = {pid: 0.5 for pid in self.positions}
        else:
            # Convert to 0-1 scale
            min_vote = min(self.votes.values())
            max_vote = max(self.votes.values())
            vote_range = max_vote - min_vote if max_vote != min_vote else 1
            scores = {
                pid: (v - min_vote) / vote_range
                for pid, v in self.votes.items()
            }

        # Find winner
        winner_id = max(self.votes, key=self.votes.get)

        # Calculate consensus strength (how concentrated are the votes?)
        vote_values = list(self.votes.values())
        if len(vote_values) > 1 and max(vote_values) > 0:
            # Herfindahl-like concentration
            total = sum(max(0, v) for v in vote_values)
            if total > 0:
                shares = [max(0, v)/total for v in vote_values]
                consensus = sum(s**2 for s in shares)  # 1/n for even split, 1 for unanimous
            else:
                consensus = 0.0
        else:
            consensus = 1.0

        # Agent contributions
        contributions = {}
        for aid, agent in self.agents.items():
            total_votes = sum(abs(v) for v in agent.votes_cast.values())
            total_spent = self.default_credits - agent.budget
            contributions[aid] = {
                "total_votes": total_votes,
                "credits_spent": total_spent,
                "votes_by_position": dict(agent.votes_cast),
            }

        return DeliberationResult(
            winning_position=self.positions[winner_id],
            position_scores=scores,
            agent_contributions=contributions,
            consensus_strength=consensus,
            rounds_completed=1,
            mechanism="quadratic_voting",
            metadata={"raw_votes": dict(self.votes)}
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION MARKET
# ═══════════════════════════════════════════════════════════════════════════════

class PredictionMarket:
    """
    Prediction market for belief aggregation.

    Uses Logarithmic Market Scoring Rule (LMSR) for automated market making.
    Agents buy/sell shares in positions, prices reflect collective belief.
    """

    def __init__(
        self,
        positions: List[Position],
        initial_liquidity: float = 100.0,
        initial_price: float = 0.5
    ):
        self.positions = {p.id: p for p in positions}
        self.liquidity = initial_liquidity  # LMSR b parameter

        # Initialize quantities (LMSR state)
        self.quantities: Dict[str, float] = {p.id: 0.0 for p in positions}

        self.agents: Dict[str, AgentState] = {}
        self.market = MarketState(
            prices={p.id: initial_price for p in positions},
            liquidity={p.id: initial_liquidity for p in positions}
        )

    def register_agent(self, agent_id: str, budget: float = 100.0):
        """Register an agent with trading budget."""
        self.agents[agent_id] = AgentState(
            agent_id=agent_id,
            budget=budget
        )

    def _cost_function(self, quantities: Dict[str, float]) -> float:
        """LMSR cost function: C(q) = b * log(sum(exp(q_i/b)))"""
        b = self.liquidity
        exp_sum = sum(math.exp(q/b) for q in quantities.values())
        return b * math.log(exp_sum)

    def _price(self, position_id: str) -> float:
        """Current price for a position: exp(q_i/b) / sum(exp(q_j/b))"""
        b = self.liquidity
        exp_qi = math.exp(self.quantities[position_id] / b)
        exp_sum = sum(math.exp(q/b) for q in self.quantities.values())
        return exp_qi / exp_sum

    def get_prices(self) -> Dict[str, float]:
        """Get current prices for all positions."""
        return {pid: self._price(pid) for pid in self.positions}

    def quote(self, position_id: str, shares: float) -> float:
        """
        Get cost to buy/sell shares.

        Args:
            position_id: Position to trade
            shares: Positive to buy, negative to sell

        Returns:
            Cost (positive) or revenue (negative)
        """
        if position_id not in self.positions:
            raise ValueError(f"Unknown position: {position_id}")

        # Current cost
        current_cost = self._cost_function(self.quantities)

        # New quantities after trade
        new_quantities = dict(self.quantities)
        new_quantities[position_id] += shares

        # New cost
        new_cost = self._cost_function(new_quantities)

        return new_cost - current_cost

    def trade(
        self,
        agent_id: str,
        position_id: str,
        shares: float
    ) -> Tuple[bool, str, float]:
        """
        Execute a trade.

        Returns:
            (success, message, cost)
        """
        if agent_id not in self.agents:
            return False, f"Agent {agent_id} not registered", 0.0

        agent = self.agents[agent_id]

        try:
            cost = self.quote(position_id, shares)
        except ValueError as e:
            return False, str(e), 0.0

        # Check budget for buys
        if cost > 0 and cost > agent.budget:
            max_shares = self._max_shares_for_budget(position_id, agent.budget)
            return False, f"Insufficient funds. Max shares: {max_shares:.2f}", 0.0

        # Execute trade
        agent.budget -= cost
        self.quantities[position_id] += shares

        # Record trade
        trade_record = {
            "agent_id": agent_id,
            "position_id": position_id,
            "shares": shares,
            "cost": cost,
            "price": self._price(position_id),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        agent.trades.append(trade_record)
        self.market.trades.append(trade_record)
        self.market.volume += abs(cost)

        # Update market prices
        self.market.prices = self.get_prices()

        return True, f"Traded {shares:.2f} shares at {cost:.2f}", cost

    def _max_shares_for_budget(self, position_id: str, budget: float) -> float:
        """Binary search for max shares affordable."""
        if budget <= 0:
            return 0.0

        low, high = 0.0, budget * 10  # Rough upper bound

        for _ in range(50):  # Binary search iterations
            mid = (low + high) / 2
            cost = self.quote(position_id, mid)

            if abs(cost - budget) < 0.01:
                return mid
            elif cost < budget:
                low = mid
            else:
                high = mid

        return low

    def get_results(self) -> DeliberationResult:
        """Get market-based results."""
        prices = self.get_prices()

        # Winner is highest priced position
        winner_id = max(prices, key=prices.get)

        # Consensus is inverse of entropy (normalized)
        entropy = -sum(p * math.log(p + 1e-10) for p in prices.values())
        max_entropy = math.log(len(prices))
        consensus = 1 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        # Agent contributions
        contributions = {}
        for aid, agent in self.agents.items():
            total_traded = sum(abs(t["cost"]) for t in agent.trades)
            net_position = {}
            for t in agent.trades:
                pid = t["position_id"]
                net_position[pid] = net_position.get(pid, 0) + t["shares"]

            contributions[aid] = {
                "total_traded": total_traded,
                "net_positions": net_position,
                "remaining_budget": agent.budget,
            }

        return DeliberationResult(
            winning_position=self.positions[winner_id],
            position_scores=prices,
            agent_contributions=contributions,
            consensus_strength=consensus,
            rounds_completed=len(self.market.trades),
            mechanism="prediction_market",
            metadata={
                "total_volume": self.market.volume,
                "final_quantities": dict(self.quantities),
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVICTION STAKING
# ═══════════════════════════════════════════════════════════════════════════════

class ConvictionStaking:
    """
    Conviction staking mechanism.

    Agents stake reputation on positions. Correct predictions increase
    reputation, incorrect ones decrease it. Final weights are
    reputation-weighted convictions.
    """

    def __init__(self, positions: List[Position]):
        self.positions = {p.id: p for p in positions}
        self.agents: Dict[str, AgentState] = {}
        self.stakes: Dict[str, Dict[str, float]] = {p.id: {} for p in positions}
        self.round = 0

    def register_agent(
        self,
        agent_id: str,
        initial_reputation: float = 1.0
    ):
        """Register agent with reputation."""
        self.agents[agent_id] = AgentState(
            agent_id=agent_id,
            reputation=initial_reputation,
            budget=1.0  # Budget is normalized conviction allocation
        )

    def stake(
        self,
        agent_id: str,
        position_id: str,
        conviction: float  # 0-1, fraction of reputation to stake
    ) -> Tuple[bool, str]:
        """
        Stake conviction on a position.

        Args:
            agent_id: Staking agent
            position_id: Position to back
            conviction: Fraction of available reputation to stake (0-1)
        """
        if agent_id not in self.agents:
            return False, "Agent not registered"

        if position_id not in self.positions:
            return False, "Position not found"

        if not 0 <= conviction <= 1:
            return False, "Conviction must be between 0 and 1"

        agent = self.agents[agent_id]

        # Check available budget
        total_staked = sum(
            self.stakes[pid].get(agent_id, 0)
            for pid in self.positions
        )

        available = 1.0 - total_staked
        if conviction > available + 0.001:  # Small epsilon for float errors
            return False, f"Insufficient budget. Available: {available:.2f}"

        # Record stake
        self.stakes[position_id][agent_id] = conviction

        return True, f"Staked {conviction:.2%} conviction on {position_id}"

    def get_weighted_scores(self) -> Dict[str, float]:
        """Get reputation-weighted conviction scores."""
        scores = {}

        for pid in self.positions:
            weighted_sum = 0.0
            for aid, conviction in self.stakes[pid].items():
                agent = self.agents[aid]
                weighted_sum += conviction * agent.reputation
            scores[pid] = weighted_sum

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {pid: s/total for pid, s in scores.items()}

        return scores

    def resolve(self, winning_position_id: str):
        """
        Resolve the deliberation and update reputations.

        Agents who staked on the winner gain reputation proportional
        to their conviction. Others lose proportionally.
        """
        if winning_position_id not in self.positions:
            raise ValueError("Invalid winning position")

        for aid, agent in self.agents.items():
            winner_stake = self.stakes[winning_position_id].get(aid, 0)

            # Calculate accuracy: how much did they stake on winner?
            total_stake = sum(
                self.stakes[pid].get(aid, 0) for pid in self.positions
            )

            if total_stake > 0:
                accuracy = winner_stake / total_stake
            else:
                accuracy = 1.0 / len(self.positions)  # No stake = random

            # Update reputation
            agent.update_reputation(accuracy)

        self.round += 1

    def get_results(self) -> DeliberationResult:
        """Get staking-based results."""
        scores = self.get_weighted_scores()

        if not scores:
            return DeliberationResult(
                winning_position=None,
                position_scores={},
                agent_contributions={},
                consensus_strength=0.0,
                rounds_completed=self.round,
                mechanism="conviction_staking"
            )

        winner_id = max(scores, key=scores.get)

        # Consensus from score concentration
        max_score = max(scores.values())
        consensus = max_score  # Winner's share of weighted conviction

        # Contributions
        contributions = {}
        for aid, agent in self.agents.items():
            stakes_by_pos = {
                pid: self.stakes[pid].get(aid, 0)
                for pid in self.positions
            }
            contributions[aid] = {
                "reputation": agent.reputation,
                "stakes": stakes_by_pos,
                "backed_winner": self.stakes[winner_id].get(aid, 0) > 0,
            }

        return DeliberationResult(
            winning_position=self.positions[winner_id],
            position_scores=scores,
            agent_contributions=contributions,
            consensus_strength=consensus,
            rounds_completed=self.round,
            mechanism="conviction_staking",
            metadata={
                "total_reputation": sum(a.reputation for a in self.agents.values())
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-ROUND DELIBERATION WITH BELIEF UPDATES
# ═══════════════════════════════════════════════════════════════════════════════

class IterativeDeliberation:
    """
    Multi-round deliberation with Bayesian belief updates.

    Agents:
    1. State initial beliefs
    2. Hear arguments from others
    3. Update beliefs based on evidence quality and source reputation
    4. Repeat until convergence or max rounds
    """

    def __init__(
        self,
        positions: List[Position],
        convergence_threshold: float = 0.05,
        max_rounds: int = 5
    ):
        self.positions = {p.id: p for p in positions}
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        self.agents: Dict[str, AgentState] = {}
        self.round_history: List[Dict] = []
        self.current_round = 0

    def register_agent(
        self,
        agent_id: str,
        initial_beliefs: Dict[str, float],
        reputation: float = 1.0
    ):
        """Register agent with initial belief distribution."""
        beliefs = {}
        for pid, prob in initial_beliefs.items():
            if pid in self.positions:
                beliefs[pid] = Belief(
                    position=self.positions[pid],
                    probability=prob,
                    confidence=0.5  # Initial moderate confidence
                )

        self.agents[agent_id] = AgentState(
            agent_id=agent_id,
            reputation=reputation,
            beliefs=beliefs
        )

    def submit_argument(
        self,
        agent_id: str,
        position_id: str,
        argument: str,
        evidence_strength: float = 0.5  # 0-1
    ):
        """Submit an argument for a position."""
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]
        if position_id in agent.beliefs:
            agent.beliefs[position_id].reasoning = argument
            agent.beliefs[position_id].confidence = min(1.0,
                agent.beliefs[position_id].confidence + evidence_strength * 0.2
            )

    def bayesian_update(
        self,
        agent_id: str,
        observed_beliefs: Dict[str, Dict[str, float]],  # other_agent -> position -> prob
        argument_quality: Dict[str, float]  # other_agent -> quality score
    ):
        """
        Update agent's beliefs based on others' positions.

        Uses reputation-weighted likelihood updating.
        """
        agent = self.agents[agent_id]

        for pid in self.positions:
            if pid not in agent.beliefs:
                continue

            prior = agent.beliefs[pid].probability

            # Aggregate evidence from others
            log_likelihood_ratio = 0.0

            for other_id, other_beliefs in observed_beliefs.items():
                if other_id == agent_id:
                    continue

                other_agent = self.agents.get(other_id)
                if not other_agent:
                    continue

                other_prob = other_beliefs.get(pid, 0.5)
                quality = argument_quality.get(other_id, 0.5)

                # Weight by reputation and argument quality
                weight = other_agent.reputation * quality

                # Log likelihood ratio
                if 0.01 < other_prob < 0.99:
                    llr = weight * math.log(other_prob / (1 - other_prob))
                    log_likelihood_ratio += llr * 0.1  # Dampening factor

            # Apply update
            prior_odds = prior / (1 - prior + 1e-10)
            posterior_odds = prior_odds * math.exp(log_likelihood_ratio)
            posterior = posterior_odds / (1 + posterior_odds)

            # Bound to avoid extremes
            posterior = max(0.01, min(0.99, posterior))

            agent.beliefs[pid].probability = posterior

    def check_convergence(self) -> bool:
        """Check if beliefs have converged."""
        if len(self.round_history) < 2:
            return False

        prev_beliefs = self.round_history[-2]
        curr_beliefs = self.round_history[-1]

        max_change = 0.0
        for aid in self.agents:
            for pid in self.positions:
                prev = prev_beliefs.get(aid, {}).get(pid, 0.5)
                curr = curr_beliefs.get(aid, {}).get(pid, 0.5)
                max_change = max(max_change, abs(curr - prev))

        return max_change < self.convergence_threshold

    def record_round(self):
        """Record current belief state."""
        state = {}
        for aid, agent in self.agents.items():
            state[aid] = {
                pid: b.probability
                for pid, b in agent.beliefs.items()
            }
        self.round_history.append(state)
        self.current_round += 1

    def get_aggregate_beliefs(self) -> Dict[str, float]:
        """Get reputation-weighted aggregate beliefs."""
        aggregates = {pid: 0.0 for pid in self.positions}
        total_rep = sum(a.reputation for a in self.agents.values())

        for agent in self.agents.values():
            weight = agent.reputation / total_rep if total_rep > 0 else 1/len(self.agents)
            for pid, belief in agent.beliefs.items():
                aggregates[pid] += belief.probability * weight

        return aggregates

    def get_results(self) -> DeliberationResult:
        """Get deliberation results."""
        aggregates = self.get_aggregate_beliefs()

        if not aggregates:
            return DeliberationResult(
                winning_position=None,
                position_scores={},
                agent_contributions={},
                consensus_strength=0.0,
                rounds_completed=self.current_round,
                mechanism="iterative_bayesian"
            )

        winner_id = max(aggregates, key=aggregates.get)

        # Consensus from belief alignment
        # Calculate variance in beliefs for winner
        winner_beliefs = [
            agent.beliefs[winner_id].probability
            for agent in self.agents.values()
            if winner_id in agent.beliefs
        ]

        if winner_beliefs:
            mean_belief = sum(winner_beliefs) / len(winner_beliefs)
            variance = sum((b - mean_belief)**2 for b in winner_beliefs) / len(winner_beliefs)
            consensus = 1 - min(1, variance * 4)  # Scale variance to 0-1
        else:
            consensus = 0.0

        contributions = {}
        for aid, agent in self.agents.items():
            contributions[aid] = {
                "final_beliefs": {pid: b.probability for pid, b in agent.beliefs.items()},
                "reputation": agent.reputation,
                "confidence": {pid: b.confidence for pid, b in agent.beliefs.items()},
            }

        return DeliberationResult(
            winning_position=self.positions[winner_id],
            position_scores=aggregates,
            agent_contributions=contributions,
            consensus_strength=consensus,
            rounds_completed=self.current_round,
            mechanism="iterative_bayesian",
            metadata={
                "belief_history": self.round_history,
                "converged": self.check_convergence(),
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MECHANISM FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

class MechanismType(Enum):
    QUADRATIC_VOTING = "quadratic_voting"
    PREDICTION_MARKET = "prediction_market"
    CONVICTION_STAKING = "conviction_staking"
    ITERATIVE_BAYESIAN = "iterative_bayesian"


def create_mechanism(
    mechanism_type: MechanismType,
    positions: List[Position],
    **kwargs
) -> Any:
    """Factory for creating deliberation mechanisms."""

    if mechanism_type == MechanismType.QUADRATIC_VOTING:
        return QuadraticVoting(
            positions,
            voice_credits=kwargs.get("voice_credits", 100.0)
        )

    elif mechanism_type == MechanismType.PREDICTION_MARKET:
        return PredictionMarket(
            positions,
            initial_liquidity=kwargs.get("liquidity", 100.0),
            initial_price=kwargs.get("initial_price", 1.0/len(positions))
        )

    elif mechanism_type == MechanismType.CONVICTION_STAKING:
        return ConvictionStaking(positions)

    elif mechanism_type == MechanismType.ITERATIVE_BAYESIAN:
        return IterativeDeliberation(
            positions,
            convergence_threshold=kwargs.get("convergence_threshold", 0.05),
            max_rounds=kwargs.get("max_rounds", 5)
        )

    else:
        raise ValueError(f"Unknown mechanism: {mechanism_type}")
