"""Identity Quorums with Forced Relationships.

Agents have distinct identities and must interact based on relationship constraints.
Relationships define who must consult whom before finalizing positions.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable

from src.llm import OpenRouterLM, Identity, IdentityWrapper, create_identity_from_prompt


class RelationshipType(Enum):
    """Types of forced relationships between agents."""
    MUST_CONSULT = "must_consult"  # A must get input from B before deciding
    MUST_OPPOSE = "must_oppose"    # A must consider counterarguments from B
    MUST_ALIGN = "must_align"      # A and B must find common ground
    VETO_POWER = "veto_power"      # B can veto A's position
    DELEGATION = "delegation"      # A delegates to B on specific topics


@dataclass
class Relationship:
    """Defines a forced relationship between two agents."""
    source: str  # Agent who has the constraint
    target: str  # Agent they must interact with
    rel_type: RelationshipType
    condition: Optional[str] = None  # When this relationship activates
    strength: float = 1.0  # How binding (0-1)
    topics: List[str] = field(default_factory=list)  # Specific topics this applies to


@dataclass
class QuorumMember:
    """A member of an identity quorum."""
    agent_id: str
    identity: Identity
    role: str  # e.g., "advocate", "critic", "synthesizer"
    expertise: List[str] = field(default_factory=list)
    influence_weight: float = 1.0

    # State
    current_position: Optional[str] = None
    position_confidence: float = 0.0
    interactions: List[Dict[str, Any]] = field(default_factory=list)
    vetoes_used: int = 0
    max_vetoes: int = 1


@dataclass
class InteractionResult:
    """Result of a forced interaction."""
    source: str
    target: str
    rel_type: RelationshipType
    source_position_before: Optional[str]
    source_position_after: Optional[str]
    position_changed: bool
    exchange: List[Dict[str, str]]  # The actual dialogue
    consensus_reached: bool = False
    veto_exercised: bool = False


class IdentityQuorum:
    """A quorum of agents with forced relationship constraints."""

    def __init__(
        self,
        topic: str,
        positions: List[Dict[str, str]],
        quorum_threshold: float = 0.67,  # Fraction needed for consensus
        model: str = "openai/gpt-4.1-mini",  # Jan 2026 default
    ):
        self.topic = topic
        self.positions = {p["id"]: p for p in positions}
        self.quorum_threshold = quorum_threshold
        self.model = model

        self.members: Dict[str, QuorumMember] = {}
        self.relationships: List[Relationship] = []
        self.interaction_log: List[InteractionResult] = []
        self.round_number = 0

    def add_member(
        self,
        agent_id: str,
        persona: str,
        role: str = "member",
        expertise: List[str] = None,
        influence_weight: float = 1.0,
    ) -> QuorumMember:
        """Add a member to the quorum."""
        identity = create_identity_from_prompt(persona)
        member = QuorumMember(
            agent_id=agent_id,
            identity=identity,
            role=role,
            expertise=expertise or [],
            influence_weight=influence_weight,
        )
        self.members[agent_id] = member
        return member

    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: RelationshipType,
        condition: str = None,
        strength: float = 1.0,
        topics: List[str] = None,
    ):
        """Add a forced relationship constraint."""
        rel = Relationship(
            source=source,
            target=target,
            rel_type=rel_type,
            condition=condition,
            strength=strength,
            topics=topics or [],
        )
        self.relationships.append(rel)

    def get_required_interactions(self, agent_id: str) -> List[Relationship]:
        """Get all relationships where this agent must interact."""
        return [r for r in self.relationships if r.source == agent_id]

    async def _create_agent_lm(self, member: QuorumMember) -> IdentityWrapper:
        """Create an LM wrapper for a member."""
        lm = OpenRouterLM(model=self.model, enable_logprobs=True)
        return IdentityWrapper(lm, member.identity)

    async def _get_initial_position(self, member: QuorumMember) -> Tuple[str, float]:
        """Get a member's initial position on the topic."""
        lm = await self._create_agent_lm(member)

        position_list = "\n".join([
            f"- {p['id']}: {p['name']} - {p.get('description', '')}"
            for p in self.positions.values()
        ])

        prompt = f"""Topic: {self.topic}

Available positions:
{position_list}

Based on your perspective and expertise, which position do you support?
Respond with:
POSITION: <position_id>
CONFIDENCE: <0-100>
REASONING: <brief explanation>"""

        response = await lm(prompt, max_tokens=200)

        # Parse response
        position = None
        confidence = 50.0

        for line in response.text.strip().split("\n"):
            if line.startswith("POSITION:"):
                pos_text = line.replace("POSITION:", "").strip().lower()
                for pid in self.positions:
                    if pid in pos_text:
                        position = pid
                        break
            elif line.startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.replace("CONFIDENCE:", "").strip().rstrip("%"))
                except:
                    pass

        # Fallback: use logprob confidence
        if position is None:
            position = list(self.positions.keys())[0]

        return position, confidence / 100.0

    async def _execute_interaction(
        self,
        source: QuorumMember,
        target: QuorumMember,
        relationship: Relationship,
    ) -> InteractionResult:
        """Execute a forced interaction between two agents."""

        source_lm = await self._create_agent_lm(source)
        target_lm = await self._create_agent_lm(target)

        exchange = []
        position_before = source.current_position

        if relationship.rel_type == RelationshipType.MUST_CONSULT:
            # Source asks target for input
            consult_prompt = f"""You must consult with {target.agent_id} ({target.role}) before finalizing your position on: {self.topic}

Your current position: {source.current_position}
Their expertise: {', '.join(target.expertise) if target.expertise else 'general'}

Ask them a specific question about the topic to inform your decision."""

            question = await source_lm(consult_prompt, max_tokens=150)
            exchange.append({"role": source.agent_id, "content": question.text})

            answer_prompt = f"""{source.agent_id} asks you: {question.text}

Topic: {self.topic}
Your position: {target.current_position}

Provide your expert input."""

            answer = await target_lm(answer_prompt, max_tokens=200)
            exchange.append({"role": target.agent_id, "content": answer.text})

            # Source reconsiders
            reconsider_prompt = f"""Based on {target.agent_id}'s input: "{answer.text}"

Your current position: {source.current_position}

Do you want to:
1. MAINTAIN your position
2. CHANGE to a different position
3. MODIFY your confidence

Respond with: DECISION: <MAINTAIN/CHANGE> and NEW_POSITION: <position_id> if changing"""

            decision = await source_lm(reconsider_prompt, max_tokens=100)
            exchange.append({"role": source.agent_id + "_decision", "content": decision.text})

        elif relationship.rel_type == RelationshipType.MUST_OPPOSE:
            # Target must provide counterargument
            oppose_prompt = f"""You must challenge {source.agent_id}'s position on: {self.topic}

Their position: {source.current_position}
Your position: {target.current_position}

Provide your strongest counterargument to their position."""

            counter = await target_lm(oppose_prompt, max_tokens=200)
            exchange.append({"role": target.agent_id, "content": counter.text})

            # Source must respond
            defend_prompt = f"""{target.agent_id} challenges your position with: "{counter.text}"

Your position: {source.current_position}

Defend your position or acknowledge valid points and potentially adjust.
Respond with your updated stance."""

            defense = await source_lm(defend_prompt, max_tokens=200)
            exchange.append({"role": source.agent_id, "content": defense.text})

        elif relationship.rel_type == RelationshipType.MUST_ALIGN:
            # Both must find common ground
            align_prompt_source = f"""You must find common ground with {target.agent_id} on: {self.topic}

Your position: {source.current_position}
Their position: {target.current_position}

Propose a synthesis or compromise."""

            proposal = await source_lm(align_prompt_source, max_tokens=200)
            exchange.append({"role": source.agent_id, "content": proposal.text})

            align_prompt_target = f"""{source.agent_id} proposes: "{proposal.text}"

Your position: {target.current_position}

Accept, modify, or counter-propose."""

            response = await target_lm(align_prompt_target, max_tokens=200)
            exchange.append({"role": target.agent_id, "content": response.text})

        elif relationship.rel_type == RelationshipType.VETO_POWER:
            # Target can veto source's position
            if target.vetoes_used < target.max_vetoes:
                veto_prompt = f"""You have VETO POWER over {source.agent_id}'s position.

Topic: {self.topic}
Their position: {source.current_position}
Your position: {target.current_position}

Do you exercise your veto? This forces them to reconsider.
Respond: VETO or ALLOW
If VETO, explain why."""

                veto_decision = await target_lm(veto_prompt, max_tokens=150)
                exchange.append({"role": target.agent_id, "content": veto_decision.text})

                if "VETO" in veto_decision.text.upper():
                    target.vetoes_used += 1

                    # Source must change
                    forced_change = f"""Your position has been VETOED by {target.agent_id}.

Reason: {veto_decision.text}

You must choose a different position from: {list(self.positions.keys())}
Current (vetoed) position: {source.current_position}

Choose a new position."""

                    new_pos = await source_lm(forced_change, max_tokens=100)
                    exchange.append({"role": source.agent_id, "content": new_pos.text})

                    return InteractionResult(
                        source=source.agent_id,
                        target=target.agent_id,
                        rel_type=relationship.rel_type,
                        source_position_before=position_before,
                        source_position_after=self._parse_position(new_pos.text),
                        position_changed=True,
                        exchange=exchange,
                        veto_exercised=True,
                    )

        elif relationship.rel_type == RelationshipType.DELEGATION:
            # Source defers to target on this topic
            delegate_prompt = f"""You are delegating the decision on "{self.topic}" to {target.agent_id}.

Their expertise: {', '.join(target.expertise)}
Their position: {target.current_position}

Accept their position as your own, or request clarification."""

            delegation = await source_lm(delegate_prompt, max_tokens=100)
            exchange.append({"role": source.agent_id, "content": delegation.text})

            # Source adopts target's position
            return InteractionResult(
                source=source.agent_id,
                target=target.agent_id,
                rel_type=relationship.rel_type,
                source_position_before=position_before,
                source_position_after=target.current_position,
                position_changed=position_before != target.current_position,
                exchange=exchange,
            )

        # Determine if position changed
        position_after = self._extract_new_position(exchange, source.current_position)

        return InteractionResult(
            source=source.agent_id,
            target=target.agent_id,
            rel_type=relationship.rel_type,
            source_position_before=position_before,
            source_position_after=position_after,
            position_changed=position_before != position_after,
            exchange=exchange,
        )

    def _parse_position(self, text: str) -> Optional[str]:
        """Parse a position ID from text."""
        text_lower = text.lower()
        for pid in self.positions:
            if pid in text_lower:
                return pid
        return None

    def _extract_new_position(self, exchange: List[Dict], current: str) -> str:
        """Extract new position from exchange, default to current."""
        # Look for explicit position changes in the exchange
        for msg in reversed(exchange):
            if "CHANGE" in msg["content"].upper():
                new_pos = self._parse_position(msg["content"])
                if new_pos:
                    return new_pos
        return current

    async def run_round(self) -> Dict[str, Any]:
        """Run one round of deliberation with forced interactions."""
        self.round_number += 1
        round_results = {
            "round": self.round_number,
            "interactions": [],
            "position_changes": [],
        }

        # Execute all required interactions
        for rel in self.relationships:
            if rel.source in self.members and rel.target in self.members:
                source = self.members[rel.source]
                target = self.members[rel.target]

                result = await self._execute_interaction(source, target, rel)
                self.interaction_log.append(result)
                round_results["interactions"].append({
                    "source": result.source,
                    "target": result.target,
                    "type": result.rel_type.value,
                    "changed": result.position_changed,
                    "veto": result.veto_exercised,
                })

                # Update source's position
                if result.position_changed:
                    source.current_position = result.source_position_after
                    round_results["position_changes"].append({
                        "agent": source.agent_id,
                        "from": result.source_position_before,
                        "to": result.source_position_after,
                    })

        return round_results

    def check_quorum(self) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """Check if quorum is reached on any position."""
        position_counts: Dict[str, float] = {pid: 0.0 for pid in self.positions}
        total_weight = sum(m.influence_weight for m in self.members.values())

        for member in self.members.values():
            if member.current_position in position_counts:
                position_counts[member.current_position] += member.influence_weight

        # Normalize
        position_fractions = {
            pid: count / total_weight if total_weight > 0 else 0
            for pid, count in position_counts.items()
        }

        # Check for quorum
        for pid, fraction in position_fractions.items():
            if fraction >= self.quorum_threshold:
                return True, pid, position_fractions

        return False, None, position_fractions

    async def deliberate(
        self,
        max_rounds: int = 5,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run full deliberation until quorum or max rounds."""

        if verbose:
            print(f"=== Identity Quorum Deliberation ===")
            print(f"Topic: {self.topic}")
            print(f"Members: {list(self.members.keys())}")
            print(f"Quorum threshold: {self.quorum_threshold:.0%}")
            print()

        # Get initial positions
        if verbose:
            print("Getting initial positions...")

        for agent_id, member in self.members.items():
            position, confidence = await self._get_initial_position(member)
            member.current_position = position
            member.position_confidence = confidence
            if verbose:
                print(f"  {agent_id}: {position} (confidence: {confidence:.0%})")

        print()

        # Check initial quorum
        has_quorum, winner, fractions = self.check_quorum()
        if has_quorum:
            if verbose:
                print(f"Immediate quorum on: {winner}")
            return {
                "quorum_reached": True,
                "winner": winner,
                "rounds": 0,
                "final_fractions": fractions,
                "interactions": [],
            }

        # Run deliberation rounds
        all_round_results = []

        for round_num in range(max_rounds):
            if verbose:
                print(f"--- Round {round_num + 1} ---")

            round_result = await self.run_round()
            all_round_results.append(round_result)

            if verbose:
                for interaction in round_result["interactions"]:
                    status = "VETO" if interaction["veto"] else ("CHANGED" if interaction["changed"] else "maintained")
                    print(f"  {interaction['source']} --[{interaction['type']}]--> {interaction['target']}: {status}")

                if round_result["position_changes"]:
                    print("  Position changes:")
                    for change in round_result["position_changes"]:
                        print(f"    {change['agent']}: {change['from']} -> {change['to']}")

            # Check quorum
            has_quorum, winner, fractions = self.check_quorum()

            if verbose:
                print(f"  Current standings: {fractions}")

            if has_quorum:
                if verbose:
                    print(f"\nQUORUM REACHED: {winner}")
                return {
                    "quorum_reached": True,
                    "winner": winner,
                    "rounds": round_num + 1,
                    "final_fractions": fractions,
                    "interactions": all_round_results,
                    "interaction_log": [
                        {
                            "source": r.source,
                            "target": r.target,
                            "type": r.rel_type.value,
                            "exchange": r.exchange,
                        }
                        for r in self.interaction_log
                    ],
                }

            print()

        # No quorum reached
        _, _, final_fractions = self.check_quorum()
        winner = max(final_fractions, key=final_fractions.get)

        if verbose:
            print(f"\nNo quorum after {max_rounds} rounds. Plurality winner: {winner}")

        return {
            "quorum_reached": False,
            "winner": winner,
            "rounds": max_rounds,
            "final_fractions": final_fractions,
            "interactions": all_round_results,
        }


# Preset quorum configurations
def create_adversarial_quorum(
    topic: str,
    positions: List[Dict[str, str]],
    model: str = "openai/gpt-4.1-mini",
) -> IdentityQuorum:
    """Create a quorum with adversarial forced relationships."""
    quorum = IdentityQuorum(topic, positions, model=model)

    # Add members with opposing viewpoints
    quorum.add_member(
        "advocate",
        "You strongly believe in progress and change. You advocate for bold action.",
        role="advocate",
        expertise=["innovation", "progress"],
        influence_weight=1.0,
    )

    quorum.add_member(
        "critic",
        "You are skeptical of change. You find flaws and risks in proposals.",
        role="critic",
        expertise=["risk analysis", "criticism"],
        influence_weight=1.0,
    )

    quorum.add_member(
        "pragmatist",
        "You seek practical, implementable solutions. You value feasibility.",
        role="pragmatist",
        expertise=["implementation", "pragmatics"],
        influence_weight=1.0,
    )

    # Forced relationships
    quorum.add_relationship("advocate", "critic", RelationshipType.MUST_OPPOSE)
    quorum.add_relationship("critic", "advocate", RelationshipType.MUST_OPPOSE)
    quorum.add_relationship("pragmatist", "advocate", RelationshipType.MUST_CONSULT)
    quorum.add_relationship("pragmatist", "critic", RelationshipType.MUST_CONSULT)

    return quorum


def create_hierarchical_quorum(
    topic: str,
    positions: List[Dict[str, str]],
    model: str = "openai/gpt-4.1-mini",
) -> IdentityQuorum:
    """Create a quorum with hierarchical authority."""
    quorum = IdentityQuorum(topic, positions, model=model, quorum_threshold=0.5)

    quorum.add_member(
        "leader",
        "You are the decision-maker. You weigh all inputs and make final calls.",
        role="leader",
        expertise=["leadership", "decision-making"],
        influence_weight=2.0,  # Double weight
    )

    quorum.add_member(
        "expert_a",
        "You are a domain expert. You provide technical analysis.",
        role="expert",
        expertise=["technical", "analysis"],
        influence_weight=1.0,
    )

    quorum.add_member(
        "expert_b",
        "You represent stakeholder interests. You advocate for affected parties.",
        role="stakeholder",
        expertise=["stakeholders", "impact"],
        influence_weight=1.0,
    )

    # Leader must consult experts
    quorum.add_relationship("leader", "expert_a", RelationshipType.MUST_CONSULT)
    quorum.add_relationship("leader", "expert_b", RelationshipType.MUST_CONSULT)

    # Experts can veto leader (once each)
    quorum.add_relationship("expert_a", "leader", RelationshipType.VETO_POWER, strength=0.5)
    quorum.add_relationship("expert_b", "leader", RelationshipType.VETO_POWER, strength=0.5)

    return quorum


def create_consensus_quorum(
    topic: str,
    positions: List[Dict[str, str]],
    model: str = "openai/gpt-4.1-mini",
) -> IdentityQuorum:
    """Create a quorum that requires full alignment."""
    quorum = IdentityQuorum(topic, positions, model=model, quorum_threshold=1.0)

    members = [
        ("member_a", "You value stability and proven approaches."),
        ("member_b", "You value innovation and new ideas."),
        ("member_c", "You value efficiency and resource optimization."),
    ]

    for mid, persona in members:
        quorum.add_member(mid, persona, role="member", influence_weight=1.0)

    # Everyone must align with everyone (round robin)
    for i, (mid1, _) in enumerate(members):
        for mid2, _ in members[i+1:]:
            quorum.add_relationship(mid1, mid2, RelationshipType.MUST_ALIGN)

    return quorum
