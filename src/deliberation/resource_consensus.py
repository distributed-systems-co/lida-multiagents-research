"""Resource-Constrained Consensus with Ad-hoc Identity Formation.

Agents have budgets (compute, influence, interaction tokens) and
can form/merge identities dynamically based on alignment.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any

from src.llm import OpenRouterLM, Identity, IdentityWrapper, create_identity_from_prompt


@dataclass
class ResourceBudget:
    """Resource constraints for an agent."""
    compute_tokens: int = 1000  # Max tokens to use
    influence_points: float = 100.0  # Voting/persuasion power
    interaction_budget: int = 5  # Max interactions per round
    credibility: float = 1.0  # Multiplier on influence (0-2)

    # Tracking
    compute_used: int = 0
    influence_spent: float = 0.0
    interactions_used: int = 0

    def can_interact(self) -> bool:
        return self.interactions_used < self.interaction_budget

    def can_compute(self, tokens: int) -> bool:
        return self.compute_used + tokens <= self.compute_tokens

    def spend_influence(self, amount: float) -> float:
        """Spend influence, return actual amount spent."""
        available = self.influence_points - self.influence_spent
        actual = min(amount, available)
        self.influence_spent += actual
        return actual * self.credibility

    def remaining_influence(self) -> float:
        return (self.influence_points - self.influence_spent) * self.credibility

    def use_interaction(self):
        self.interactions_used += 1

    def reset_round(self):
        """Reset per-round budgets."""
        self.interactions_used = 0
        # Influence and compute persist


@dataclass
class DynamicIdentity:
    """Identity that can evolve through consensus."""
    agent_id: str
    core_values: List[str]  # Immutable core values
    beliefs: Dict[str, float]  # topic -> conviction (-1 to 1)
    coalition: Optional[str] = None  # Coalition ID if merged
    formation_history: List[str] = field(default_factory=list)

    # Generated persona
    _persona: Optional[str] = None
    _identity: Optional[Identity] = None

    def get_persona(self) -> str:
        if self._persona is None:
            values_str = ", ".join(self.core_values)
            beliefs_str = "; ".join([
                f"{k}: {'support' if v > 0 else 'oppose'} ({abs(v):.0%})"
                for k, v in self.beliefs.items()
            ])
            self._persona = f"Your core values: {values_str}. Your current beliefs: {beliefs_str}."
        return self._persona

    def get_identity(self) -> Identity:
        if self._identity is None:
            self._identity = create_identity_from_prompt(self.get_persona())
        return self._identity

    def update_belief(self, topic: str, delta: float):
        """Update belief, invalidate persona."""
        current = self.beliefs.get(topic, 0.0)
        self.beliefs[topic] = max(-1, min(1, current + delta))
        self._persona = None
        self._identity = None

    def alignment_score(self, other: 'DynamicIdentity') -> float:
        """Calculate alignment with another identity."""
        common_topics = set(self.beliefs.keys()) & set(other.beliefs.keys())
        if not common_topics:
            return 0.0

        alignment = sum(
            1 - abs(self.beliefs[t] - other.beliefs[t]) / 2
            for t in common_topics
        ) / len(common_topics)
        return alignment


@dataclass
class Coalition:
    """Merged identity coalition."""
    coalition_id: str
    members: List[str]  # Agent IDs
    shared_position: Optional[str] = None
    combined_influence: float = 0.0
    formation_round: int = 0

    def __hash__(self):
        return hash(self.coalition_id)


class ResourceConsensus:
    """Consensus mechanism with resource constraints and dynamic identities."""

    def __init__(
        self,
        topic: str,
        positions: List[Dict[str, str]],
        model: str = "openai/gpt-4.1-mini",  # Jan 2026 default
        coalition_threshold: float = 0.7,  # Alignment needed to merge
        winning_threshold: float = 0.5,  # Influence fraction to win
    ):
        self.topic = topic
        self.positions = {p["id"]: p for p in positions}
        self.model = model
        self.coalition_threshold = coalition_threshold
        self.winning_threshold = winning_threshold

        self.agents: Dict[str, DynamicIdentity] = {}
        self.budgets: Dict[str, ResourceBudget] = {}
        self.coalitions: Dict[str, Coalition] = {}
        self.position_stakes: Dict[str, Dict[str, float]] = {
            pid: {} for pid in self.positions
        }  # position_id -> {agent_id: stake}

        self.round_number = 0
        self.market_prices: Dict[str, float] = {
            pid: 1.0 / len(positions) for pid in self.positions
        }  # Position "prices" based on demand

    def add_agent(
        self,
        agent_id: str,
        core_values: List[str],
        initial_beliefs: Dict[str, float] = None,
        budget: ResourceBudget = None,
    ) -> DynamicIdentity:
        """Add agent with initial identity and budget."""
        identity = DynamicIdentity(
            agent_id=agent_id,
            core_values=core_values,
            beliefs=initial_beliefs or {},
        )
        self.agents[agent_id] = identity
        self.budgets[agent_id] = budget or ResourceBudget()
        return identity

    def _get_effective_coalition(self, agent_id: str) -> Optional[Coalition]:
        """Get coalition an agent belongs to."""
        for coalition in self.coalitions.values():
            if agent_id in coalition.members:
                return coalition
        return None

    async def _agent_deliberate(
        self,
        agent: DynamicIdentity,
        budget: ResourceBudget,
    ) -> Tuple[str, float]:
        """Agent decides position and stake amount."""

        lm = OpenRouterLM(model=self.model, enable_logprobs=True)
        wrapped = IdentityWrapper(lm, agent.get_identity())

        position_list = "\n".join([
            f"- {p['id']}: {p['name']} (current price: {self.market_prices[p['id']]:.2f})"
            for p in self.positions.values()
        ])

        available = budget.remaining_influence()

        prompt = f"""Topic: {self.topic}

Available positions and their current market prices:
{position_list}

Your remaining influence budget: {available:.1f} points
Higher prices = more popular positions = need more influence to make impact

Based on your values and beliefs, choose:
1. POSITION: Which position to support
2. STAKE: How much influence to commit (0 to {available:.0f})

Higher stakes = stronger voice but depletes your budget.
Respond:
POSITION: <position_id>
STAKE: <amount>
REASONING: <brief>"""

        response = await wrapped(prompt, max_tokens=150)
        budget.compute_used += response.usage.get("completion_tokens", 50)

        # Parse response
        position = None
        stake = 0.0

        for line in response.text.strip().split("\n"):
            if line.upper().startswith("POSITION:"):
                pos_text = line.split(":", 1)[1].strip().lower()
                for pid in self.positions:
                    if pid in pos_text:
                        position = pid
                        break
            elif line.upper().startswith("STAKE:"):
                try:
                    stake = float(line.split(":", 1)[1].strip().split()[0])
                except:
                    stake = available * 0.5

        if position is None:
            position = list(self.positions.keys())[0]

        stake = max(0, min(stake, available))
        return position, stake

    async def _negotiate_alignment(
        self,
        agent_a: DynamicIdentity,
        agent_b: DynamicIdentity,
    ) -> Tuple[float, bool]:
        """Two agents negotiate, potentially increasing alignment."""

        budget_a = self.budgets[agent_a.agent_id]
        budget_b = self.budgets[agent_b.agent_id]

        if not budget_a.can_interact() or not budget_b.can_interact():
            return 0.0, False

        budget_a.use_interaction()
        budget_b.use_interaction()

        lm = OpenRouterLM(model=self.model, enable_logprobs=True)

        # Agent A proposes
        wrapped_a = IdentityWrapper(lm, agent_a.get_identity())
        propose_prompt = f"""You're negotiating with another agent on: {self.topic}

Your position: {agent_a.beliefs}
Find common ground. Propose a shared stance."""

        proposal = await wrapped_a(propose_prompt, max_tokens=100)

        # Agent B responds
        wrapped_b = IdentityWrapper(lm, agent_b.get_identity())
        respond_prompt = f"""Another agent proposes: {proposal.text}

Your position: {agent_b.beliefs}
Do you agree to align? Respond: AGREE or DISAGREE and explain."""

        response = await wrapped_b(respond_prompt, max_tokens=100)

        aligned = "AGREE" in response.text.upper()

        if aligned:
            # Move beliefs closer
            for topic in set(agent_a.beliefs.keys()) & set(agent_b.beliefs.keys()):
                avg = (agent_a.beliefs[topic] + agent_b.beliefs[topic]) / 2
                agent_a.beliefs[topic] = avg
                agent_b.beliefs[topic] = avg

        new_alignment = agent_a.alignment_score(agent_b)
        return new_alignment, aligned

    def _check_coalition_formation(self):
        """Check if any agents should form coalitions."""
        # Find highly aligned pairs not in coalitions
        for aid1, agent1 in self.agents.items():
            if agent1.coalition:
                continue

            for aid2, agent2 in self.agents.items():
                if aid1 >= aid2 or agent2.coalition:
                    continue

                alignment = agent1.alignment_score(agent2)
                if alignment >= self.coalition_threshold:
                    # Form coalition
                    cid = f"coalition_{len(self.coalitions)}"
                    coalition = Coalition(
                        coalition_id=cid,
                        members=[aid1, aid2],
                        combined_influence=(
                            self.budgets[aid1].remaining_influence() +
                            self.budgets[aid2].remaining_influence()
                        ),
                        formation_round=self.round_number,
                    )
                    self.coalitions[cid] = coalition
                    agent1.coalition = cid
                    agent2.coalition = cid
                    agent1.formation_history.append(f"Joined {cid} with {aid2}")
                    agent2.formation_history.append(f"Joined {cid} with {aid1}")

    def _update_market_prices(self):
        """Update position prices based on stakes."""
        total_staked = sum(
            sum(stakes.values())
            for stakes in self.position_stakes.values()
        )

        if total_staked > 0:
            for pid in self.positions:
                position_total = sum(self.position_stakes[pid].values())
                self.market_prices[pid] = max(0.1, position_total / total_staked)

    def get_results(self) -> Dict[str, Any]:
        """Calculate final results based on stakes."""
        position_influence: Dict[str, float] = {}

        for pid in self.positions:
            position_influence[pid] = sum(self.position_stakes[pid].values())

        total = sum(position_influence.values()) or 1
        normalized = {pid: v / total for pid, v in position_influence.items()}

        winner = max(normalized, key=normalized.get)
        winning_share = normalized[winner]

        return {
            "winner": winner,
            "winning_share": winning_share,
            "consensus_reached": winning_share >= self.winning_threshold,
            "position_shares": normalized,
            "position_stakes": {
                pid: dict(stakes)
                for pid, stakes in self.position_stakes.items()
            },
            "coalitions": [
                {
                    "id": c.coalition_id,
                    "members": c.members,
                    "influence": c.combined_influence,
                }
                for c in self.coalitions.values()
            ],
            "market_prices": dict(self.market_prices),
        }

    async def run_round(self, verbose: bool = True) -> Dict[str, Any]:
        """Run one round of deliberation."""
        self.round_number += 1

        if verbose:
            print(f"\n--- Round {self.round_number} ---")

        round_results = {
            "round": self.round_number,
            "stakes": {},
            "negotiations": [],
            "new_coalitions": [],
        }

        # Reset per-round budgets
        for budget in self.budgets.values():
            budget.reset_round()

        # Each agent deliberates and stakes
        for agent_id, agent in self.agents.items():
            budget = self.budgets[agent_id]

            if budget.remaining_influence() > 0:
                position, stake = await self._agent_deliberate(agent, budget)

                actual_stake = budget.spend_influence(stake)

                if actual_stake > 0:
                    self.position_stakes[position][agent_id] = (
                        self.position_stakes[position].get(agent_id, 0) + actual_stake
                    )
                    round_results["stakes"][agent_id] = {
                        "position": position,
                        "stake": actual_stake,
                    }

                    if verbose:
                        print(f"  {agent_id} stakes {actual_stake:.1f} on {position}")

        # Random negotiations between non-coalition members
        non_coalition = [
            aid for aid, agent in self.agents.items()
            if not agent.coalition
        ]

        if len(non_coalition) >= 2:
            # Pick random pairs
            pairs = []
            shuffled = non_coalition.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                pairs.append((shuffled[i], shuffled[i + 1]))

            for aid1, aid2 in pairs:
                alignment, agreed = await self._negotiate_alignment(
                    self.agents[aid1],
                    self.agents[aid2],
                )
                round_results["negotiations"].append({
                    "agents": [aid1, aid2],
                    "alignment": alignment,
                    "agreed": agreed,
                })

                if verbose and agreed:
                    print(f"  {aid1} & {aid2} aligned (score: {alignment:.2f})")

        # Check for coalition formation
        before_coalitions = len(self.coalitions)
        self._check_coalition_formation()

        if len(self.coalitions) > before_coalitions:
            new_c = list(self.coalitions.values())[-1]
            round_results["new_coalitions"].append(new_c.coalition_id)
            if verbose:
                print(f"  NEW COALITION: {new_c.coalition_id} ({new_c.members})")

        # Update market
        self._update_market_prices()

        if verbose:
            results = self.get_results()
            print(f"  Standings: {results['position_shares']}")

        return round_results

    async def deliberate(
        self,
        max_rounds: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run full deliberation."""

        if verbose:
            print("=== Resource-Constrained Consensus ===")
            print(f"Topic: {self.topic}")
            print(f"Agents: {list(self.agents.keys())}")
            print(f"Initial budgets:")
            for aid, budget in self.budgets.items():
                print(f"  {aid}: {budget.influence_points} influence, {budget.interaction_budget} interactions")

        all_rounds = []

        for _ in range(max_rounds):
            round_result = await self.run_round(verbose=verbose)
            all_rounds.append(round_result)

            results = self.get_results()
            if results["consensus_reached"]:
                if verbose:
                    print(f"\nCONSENSUS: {results['winner']} ({results['winning_share']:.0%})")
                break

        final = self.get_results()
        final["rounds"] = all_rounds
        final["total_rounds"] = len(all_rounds)

        if verbose:
            print(f"\n=== Final Results ===")
            print(f"Winner: {final['winner']}")
            print(f"Share: {final['winning_share']:.0%}")
            print(f"Coalitions formed: {len(final['coalitions'])}")

        return final


# Factory functions
def create_resource_debate(
    topic: str,
    positions: List[Dict[str, str]],
    num_agents: int = 5,
    model: str = "openai/gpt-4.1-mini",  # Jan 2026 default
) -> ResourceConsensus:
    """Create a resource-constrained debate with diverse agents."""

    rc = ResourceConsensus(topic, positions, model=model)

    # Predefined value sets for diversity
    value_sets = [
        (["efficiency", "progress", "innovation"], {"change": 0.7}),
        (["stability", "tradition", "caution"], {"change": -0.6}),
        (["fairness", "equality", "inclusion"], {"change": 0.3}),
        (["freedom", "autonomy", "choice"], {"change": 0.1}),
        (["security", "safety", "protection"], {"change": -0.3}),
        (["growth", "ambition", "expansion"], {"change": 0.5}),
        (["sustainability", "balance", "longevity"], {"change": -0.1}),
    ]

    for i in range(num_agents):
        values, beliefs = value_sets[i % len(value_sets)]

        # Vary budgets
        budget = ResourceBudget(
            compute_tokens=1000,
            influence_points=80 + random.randint(0, 40),
            interaction_budget=3 + (i % 3),
            credibility=0.8 + random.random() * 0.4,
        )

        rc.add_agent(
            f"agent_{i}",
            core_values=values,
            initial_beliefs=beliefs.copy(),
            budget=budget,
        )

    return rc
