"""
Information Dynamics Module

Advanced information propagation and epistemic modeling:
- Belief Propagation Networks
- Epistemic Logic and Common Knowledge
- Information Cascades
- Misinformation and Correction Dynamics
- Social Learning
- Opinion Dynamics
- Rumor Spreading Models
- Echo Chambers and Filter Bubbles

Based on:
- Epistemic Game Theory
- Social Network Analysis
- Information Theory
- Computational Social Science
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# ============================================================================
# BELIEF SYSTEMS
# ============================================================================

class BeliefType(Enum):
    """Types of beliefs"""
    FACTUAL = "factual"  # About world states
    NORMATIVE = "normative"  # About what should be
    EPISTEMIC = "epistemic"  # About what is known
    MODAL = "modal"  # About possibilities
    PROBABILISTIC = "probabilistic"  # Uncertain beliefs
    HIGHER_ORDER = "higher_order"  # Beliefs about beliefs


class BeliefSource(Enum):
    """Sources of beliefs"""
    OBSERVATION = "observation"
    TESTIMONY = "testimony"
    INFERENCE = "inference"
    MEMORY = "memory"
    INTUITION = "intuition"
    AUTHORITY = "authority"
    CONSENSUS = "consensus"


@dataclass
class Belief:
    """A belief held by an agent"""
    id: str
    proposition: str
    belief_type: BeliefType
    credence: float  # Subjective probability (0-1)
    source: BeliefSource
    evidence: List[str] = field(default_factory=list)
    acquired_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    revision_count: int = 0

    def update_credence(self, new_credence: float, reason: str):
        """Update belief credence"""
        self.credence = new_credence
        self.last_updated = datetime.now()
        self.revision_count += 1
        self.evidence.append(reason)


@dataclass
class EpistemicState:
    """Agent's complete epistemic state"""
    agent_id: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    knowledge: Set[str] = field(default_factory=set)  # Justified true beliefs
    uncertainties: Dict[str, float] = field(default_factory=dict)

    # Higher-order beliefs (beliefs about others' beliefs)
    beliefs_about_others: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Common knowledge
    common_knowledge: Set[str] = field(default_factory=set)

    def get_credence(self, proposition: str) -> float:
        """Get credence in a proposition"""
        if proposition in self.beliefs:
            return self.beliefs[proposition].credence
        return 0.5  # Uncertain


class BeliefNetwork:
    """
    Network of interconnected beliefs with dependency structure.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = EpistemicState(agent_id=agent_id)
        self.belief_graph: Dict[str, Set[str]] = defaultdict(set)  # Belief -> supporting beliefs
        self.coherence_threshold = 0.7

    def add_belief(self, belief: Belief, supports: List[str] = None):
        """Add a belief to the network"""
        self.state.beliefs[belief.id] = belief

        if supports:
            for supporting_id in supports:
                self.belief_graph[belief.id].add(supporting_id)

    def revise_belief(self, belief_id: str, new_evidence: Dict[str, Any]):
        """Revise a belief based on new evidence"""
        if belief_id not in self.state.beliefs:
            return

        belief = self.state.beliefs[belief_id]

        # Bayesian update
        prior = belief.credence
        likelihood = new_evidence.get("likelihood", 0.5)
        evidence_strength = new_evidence.get("strength", 0.5)

        # Simplified Bayes
        posterior = (prior * likelihood) / (
            prior * likelihood + (1 - prior) * (1 - likelihood)
        )

        # Weight by evidence strength
        new_credence = prior + (posterior - prior) * evidence_strength
        belief.update_credence(new_credence, str(new_evidence))

        # Propagate to dependent beliefs
        self._propagate_revision(belief_id)

    def _propagate_revision(self, revised_id: str):
        """Propagate belief revision through the network"""
        revised_credence = self.state.beliefs[revised_id].credence

        # Find beliefs that depend on this one
        for belief_id, supports in self.belief_graph.items():
            if revised_id in supports:
                if belief_id in self.state.beliefs:
                    dependent = self.state.beliefs[belief_id]
                    # Adjust credence based on support
                    support_credences = [
                        self.state.beliefs[s].credence
                        for s in supports
                        if s in self.state.beliefs
                    ]
                    if support_credences:
                        avg_support = sum(support_credences) / len(support_credences)
                        # Move toward support average
                        adjustment = (avg_support - dependent.credence) * 0.3
                        dependent.credence = max(0, min(1, dependent.credence + adjustment))

    def check_coherence(self) -> Dict[str, Any]:
        """Check coherence of belief network"""
        inconsistencies = []

        # Check for contradictions
        propositions = {}
        for belief_id, belief in self.state.beliefs.items():
            prop = belief.proposition

            if prop.startswith("NOT "):
                positive = prop[4:]
                if positive in propositions:
                    pos_credence = propositions[positive]
                    neg_credence = belief.credence
                    if pos_credence + neg_credence > 1.3:  # Should sum close to 1
                        inconsistencies.append({
                            "type": "contradiction",
                            "beliefs": [positive, prop],
                            "severity": abs((pos_credence + neg_credence) - 1)
                        })
            else:
                propositions[prop] = belief.credence

        # Calculate overall coherence
        if not self.state.beliefs:
            coherence = 1.0
        else:
            # Coherence based on support structure
            supported = sum(1 for b in self.belief_graph if self.belief_graph[b])
            coherence = supported / len(self.state.beliefs) if self.state.beliefs else 1.0

        return {
            "coherence_score": coherence,
            "inconsistencies": inconsistencies,
            "is_coherent": coherence >= self.coherence_threshold
        }


# ============================================================================
# INFORMATION PROPAGATION
# ============================================================================

class PropagationType(Enum):
    """Types of information propagation"""
    BROADCAST = "broadcast"  # One to many
    CASCADE = "cascade"  # Sequential spreading
    VIRAL = "viral"  # Exponential spreading
    DIFFUSION = "diffusion"  # Gradual spreading
    EPIDEMIC = "epidemic"  # SIR model


@dataclass
class InformationItem:
    """A piece of information that can spread"""
    id: str
    content: str
    source: str
    veracity: float  # 0-1, how true it is
    virality: float = 0.5  # How likely to spread
    credibility: float = 0.5  # How believable
    decay_rate: float = 0.01
    created_at: datetime = field(default_factory=datetime.now)

    # Spreading statistics
    exposures: int = 0
    adoptions: int = 0
    rejections: int = 0

    @property
    def adoption_rate(self) -> float:
        if self.exposures == 0:
            return 0.5
        return self.adoptions / self.exposures


@dataclass
class AgentBeliefState:
    """Agent state for information dynamics"""
    agent_id: str
    beliefs: Dict[str, float] = field(default_factory=dict)
    exposure_history: List[str] = field(default_factory=list)
    credibility_assessments: Dict[str, float] = field(default_factory=dict)

    # Susceptibility factors
    gullibility: float = 0.5
    skepticism: float = 0.5
    conformity: float = 0.5

    # Network position
    influence: float = 0.5
    connections: Set[str] = field(default_factory=set)


class InformationDynamics:
    """
    Simulates information spreading through a network.
    """

    def __init__(self):
        self.agents: Dict[str, AgentBeliefState] = {}
        self.information: Dict[str, InformationItem] = {}
        self.network: Dict[str, Set[str]] = defaultdict(set)  # Agent -> connected agents
        self.propagation_history: List[Dict[str, Any]] = []

        # Model parameters
        self.base_transmission_rate = 0.3
        self.recovery_rate = 0.1  # For SIR model
        self.truth_bias = 0.6  # Higher = truer info spreads better

    def add_agent(self, agent: AgentBeliefState):
        """Add an agent to the network"""
        self.agents[agent.agent_id] = agent
        for connection in agent.connections:
            self.network[agent.agent_id].add(connection)
            self.network[connection].add(agent.agent_id)

    def introduce_information(self, info: InformationItem) -> str:
        """Introduce new information into the network"""
        self.information[info.id] = info

        # Source agent adopts immediately
        if info.source in self.agents:
            self.agents[info.source].beliefs[info.id] = info.credibility
            info.adoptions += 1

        return info.id

    def expose_agent(self, agent_id: str, info_id: str,
                    source_agent: str = None) -> bool:
        """Expose an agent to information. Returns if adopted."""
        if agent_id not in self.agents or info_id not in self.information:
            return False

        agent = self.agents[agent_id]
        info = self.information[info_id]

        # Already exposed?
        if info_id in agent.exposure_history:
            return info_id in agent.beliefs

        agent.exposure_history.append(info_id)
        info.exposures += 1

        # Calculate adoption probability
        adoption_prob = self._calculate_adoption_probability(agent, info, source_agent)

        # Decide
        adopted = random.random() < adoption_prob

        if adopted:
            agent.beliefs[info_id] = info.credibility
            info.adoptions += 1
        else:
            info.rejections += 1

        self.propagation_history.append({
            "agent": agent_id,
            "info": info_id,
            "source": source_agent,
            "adopted": adopted,
            "probability": adoption_prob,
            "timestamp": datetime.now()
        })

        return adopted

    def _calculate_adoption_probability(self, agent: AgentBeliefState,
                                        info: InformationItem,
                                        source_agent: str = None) -> float:
        """Calculate probability an agent adopts information"""
        # Base probability from info properties
        base_prob = info.credibility * info.virality

        # Agent factors
        if agent.skepticism > 0.5:
            base_prob *= (1 - agent.skepticism * 0.5)
        else:
            base_prob *= (1 + agent.gullibility * 0.3)

        # Source credibility
        if source_agent:
            source_cred = agent.credibility_assessments.get(source_agent, 0.5)
            base_prob *= source_cred

        # Social proof (conformity)
        if agent.connections:
            believers = sum(
                1 for c in agent.connections
                if c in self.agents and info.id in self.agents[c].beliefs
            )
            social_proof = believers / len(agent.connections)
            base_prob += agent.conformity * social_proof * 0.3

        # Truth bias
        base_prob *= (1 + (info.veracity - 0.5) * self.truth_bias)

        return max(0, min(1, base_prob))

    def propagate_step(self) -> Dict[str, Any]:
        """Run one step of information propagation"""
        new_exposures = 0
        new_adoptions = 0

        for info_id, info in self.information.items():
            # Find current believers
            believers = [
                agent_id for agent_id, agent in self.agents.items()
                if info_id in agent.beliefs
            ]

            for believer_id in believers:
                believer = self.agents[believer_id]

                # Spread to connections
                for contact_id in believer.connections:
                    if contact_id in self.agents:
                        contact = self.agents[contact_id]

                        # Skip if already exposed
                        if info_id in contact.exposure_history:
                            continue

                        # Transmission probability
                        trans_prob = (
                            self.base_transmission_rate *
                            believer.influence *
                            info.virality
                        )

                        if random.random() < trans_prob:
                            new_exposures += 1
                            if self.expose_agent(contact_id, info_id, believer_id):
                                new_adoptions += 1

            # Decay information over time
            info.virality *= (1 - info.decay_rate)

        return {
            "new_exposures": new_exposures,
            "new_adoptions": new_adoptions,
            "active_information": sum(
                1 for info in self.information.values()
                if info.virality > 0.1
            )
        }

    def run_cascade(self, info_id: str, steps: int = 50) -> Dict[str, Any]:
        """Run information cascade for given steps"""
        history = []

        for step in range(steps):
            result = self.propagate_step()
            history.append(result)

            if result["new_exposures"] == 0:
                break

        info = self.information.get(info_id)
        return {
            "final_adoption_rate": info.adoption_rate if info else 0,
            "total_adopters": info.adoptions if info else 0,
            "total_exposures": info.exposures if info else 0,
            "steps_taken": len(history),
            "history": history
        }


# ============================================================================
# MISINFORMATION DYNAMICS
# ============================================================================

@dataclass
class MisinformationItem(InformationItem):
    """Misinformation with additional properties"""
    true_counterpart: Optional[str] = None  # ID of true version
    deceptiveness: float = 0.5  # How deceptive it is
    debunked: bool = False
    debunk_exposure: float = 0.0  # What fraction exposed to debunk


class MisinformationDynamics:
    """
    Models misinformation spread and correction.
    """

    def __init__(self, base_dynamics: InformationDynamics):
        self.dynamics = base_dynamics
        self.misinformation: Dict[str, MisinformationItem] = {}
        self.corrections: Dict[str, InformationItem] = {}  # Misinfo ID -> correction
        self.fact_checks: Dict[str, float] = {}  # Item ID -> fact-check score

        # Continued influence effect
        self.continued_influence = 0.7  # How much misinfo persists after correction

    def introduce_misinformation(self, misinfo: MisinformationItem) -> str:
        """Introduce misinformation"""
        self.misinformation[misinfo.id] = misinfo
        return self.dynamics.introduce_information(misinfo)

    def introduce_correction(self, misinfo_id: str, correction: InformationItem):
        """Introduce a correction for misinformation"""
        if misinfo_id not in self.misinformation:
            return

        self.corrections[misinfo_id] = correction
        self.misinformation[misinfo_id].debunked = True

        return self.dynamics.introduce_information(correction)

    def apply_correction(self, agent_id: str, misinfo_id: str) -> float:
        """Apply correction to an agent. Returns new belief level."""
        if agent_id not in self.dynamics.agents:
            return 0.0
        if misinfo_id not in self.misinformation:
            return 0.0

        agent = self.dynamics.agents[agent_id]
        misinfo = self.misinformation[misinfo_id]

        if misinfo_id not in agent.beliefs:
            return 0.0

        current_belief = agent.beliefs[misinfo_id]

        # Reduction based on agent skepticism and continued influence
        reduction = (1 - self.continued_influence) * (1 - agent.gullibility + agent.skepticism)

        new_belief = current_belief * (1 - reduction)
        agent.beliefs[misinfo_id] = new_belief

        return new_belief

    def fact_check(self, info_id: str) -> float:
        """Fact-check an information item. Returns veracity score."""
        if info_id in self.dynamics.information:
            veracity = self.dynamics.information[info_id].veracity
        elif info_id in self.misinformation:
            veracity = self.misinformation[info_id].veracity
        else:
            veracity = 0.5

        self.fact_checks[info_id] = veracity
        return veracity

    def get_misinformation_metrics(self) -> Dict[str, Any]:
        """Get metrics on misinformation spread"""
        total_misinfo = len(self.misinformation)
        debunked = sum(1 for m in self.misinformation.values() if m.debunked)

        # Calculate belief persistence
        persistence_scores = []
        for misinfo_id, misinfo in self.misinformation.items():
            if misinfo.debunked:
                believers = sum(
                    1 for agent in self.dynamics.agents.values()
                    if misinfo_id in agent.beliefs and agent.beliefs[misinfo_id] > 0.5
                )
                if misinfo.adoptions > 0:
                    persistence_scores.append(believers / misinfo.adoptions)

        return {
            "total_misinformation": total_misinfo,
            "debunked_count": debunked,
            "corrections_issued": len(self.corrections),
            "average_persistence": (
                sum(persistence_scores) / len(persistence_scores)
                if persistence_scores else 0
            ),
            "fact_checks_performed": len(self.fact_checks)
        }


# ============================================================================
# OPINION DYNAMICS
# ============================================================================

class OpinionModel(Enum):
    """Models for opinion dynamics"""
    DEGROOT = "degroot"  # Weighted average
    BOUNDED_CONFIDENCE = "bounded_confidence"  # Only influenced by similar
    VOTER = "voter"  # Copy random neighbor
    AXELROD = "axelrod"  # Cultural dissemination


@dataclass
class Opinion:
    """An opinion on a continuous scale"""
    topic: str
    position: float  # 0-1 scale
    confidence: float = 0.5
    flexibility: float = 0.5  # How easily changed


class OpinionDynamicsSimulator:
    """
    Simulates opinion dynamics in a social network.
    """

    def __init__(self, model: OpinionModel = OpinionModel.BOUNDED_CONFIDENCE):
        self.model = model
        self.agents: Dict[str, Dict[str, Opinion]] = {}  # Agent -> topic -> opinion
        self.network: Dict[str, Set[str]] = defaultdict(set)
        self.influence_weights: Dict[Tuple[str, str], float] = {}

        # Model parameters
        self.epsilon = 0.3  # Bounded confidence threshold
        self.mu = 0.5  # Convergence rate

    def add_agent(self, agent_id: str, opinions: Dict[str, Opinion],
                 connections: Set[str] = None):
        """Add an agent with opinions"""
        self.agents[agent_id] = opinions
        if connections:
            self.network[agent_id] = connections

    def set_influence_weight(self, from_agent: str, to_agent: str, weight: float):
        """Set influence weight between agents"""
        self.influence_weights[(from_agent, to_agent)] = weight

    def step(self, topic: str) -> Dict[str, float]:
        """Run one step of opinion dynamics for a topic"""
        new_opinions = {}

        for agent_id in self.agents:
            if topic not in self.agents[agent_id]:
                continue

            current = self.agents[agent_id][topic]

            if self.model == OpinionModel.DEGROOT:
                new_pos = self._degroot_update(agent_id, topic)
            elif self.model == OpinionModel.BOUNDED_CONFIDENCE:
                new_pos = self._bounded_confidence_update(agent_id, topic)
            elif self.model == OpinionModel.VOTER:
                new_pos = self._voter_update(agent_id, topic)
            else:
                new_pos = current.position

            new_opinions[agent_id] = new_pos

        # Apply updates
        for agent_id, new_pos in new_opinions.items():
            if topic in self.agents[agent_id]:
                self.agents[agent_id][topic].position = new_pos

        return new_opinions

    def _degroot_update(self, agent_id: str, topic: str) -> float:
        """DeGroot weighted average update"""
        current = self.agents[agent_id][topic].position
        neighbors = self.network.get(agent_id, set())

        if not neighbors:
            return current

        weighted_sum = current  # Self-weight
        total_weight = 1.0

        for neighbor_id in neighbors:
            if neighbor_id in self.agents and topic in self.agents[neighbor_id]:
                weight = self.influence_weights.get(
                    (neighbor_id, agent_id),
                    1.0 / (len(neighbors) + 1)
                )
                weighted_sum += weight * self.agents[neighbor_id][topic].position
                total_weight += weight

        return weighted_sum / total_weight

    def _bounded_confidence_update(self, agent_id: str, topic: str) -> float:
        """Bounded confidence update (Hegselmann-Krause)"""
        current = self.agents[agent_id][topic].position
        neighbors = self.network.get(agent_id, set())

        similar = [current]  # Include self
        for neighbor_id in neighbors:
            if neighbor_id in self.agents and topic in self.agents[neighbor_id]:
                neighbor_pos = self.agents[neighbor_id][topic].position
                if abs(neighbor_pos - current) <= self.epsilon:
                    similar.append(neighbor_pos)

        # Move toward average of similar
        avg = sum(similar) / len(similar)
        return current + self.mu * (avg - current)

    def _voter_update(self, agent_id: str, topic: str) -> float:
        """Voter model - copy random neighbor"""
        neighbors = list(self.network.get(agent_id, set()))

        if not neighbors:
            return self.agents[agent_id][topic].position

        chosen = random.choice(neighbors)
        if chosen in self.agents and topic in self.agents[chosen]:
            return self.agents[chosen][topic].position

        return self.agents[agent_id][topic].position

    def detect_polarization(self, topic: str) -> float:
        """Detect level of polarization on a topic"""
        positions = [
            self.agents[a][topic].position
            for a in self.agents
            if topic in self.agents[a]
        ]

        if len(positions) < 2:
            return 0.0

        # Variance-based polarization
        mean = sum(positions) / len(positions)
        variance = sum((p - mean) ** 2 for p in positions) / len(positions)

        # Bimodality check
        below = sum(1 for p in positions if p < 0.3)
        above = sum(1 for p in positions if p > 0.7)
        middle = len(positions) - below - above

        bimodality = (below + above - middle) / len(positions)

        return (variance + max(0, bimodality)) / 2

    def detect_echo_chambers(self) -> List[Set[str]]:
        """Detect echo chambers (clusters with similar opinions)"""
        # Simple clustering based on opinion similarity
        chambers = []
        assigned = set()

        for agent_id in self.agents:
            if agent_id in assigned:
                continue

            chamber = {agent_id}
            neighbors = list(self.network.get(agent_id, set()))

            while neighbors:
                neighbor = neighbors.pop()
                if neighbor in assigned:
                    continue

                # Check opinion similarity
                similar = True
                for topic in self.agents[agent_id]:
                    if topic in self.agents.get(neighbor, {}):
                        diff = abs(
                            self.agents[agent_id][topic].position -
                            self.agents[neighbor][topic].position
                        )
                        if diff > 0.2:
                            similar = False
                            break

                if similar:
                    chamber.add(neighbor)
                    neighbors.extend(self.network.get(neighbor, set()))

            if len(chamber) >= 3:
                chambers.append(chamber)
                assigned.update(chamber)

        return chambers


# ============================================================================
# EPISTEMIC NETWORK
# ============================================================================

class EpistemicNetwork:
    """
    Network for modeling common knowledge and distributed cognition.
    """

    def __init__(self):
        self.agents: Dict[str, BeliefNetwork] = {}
        self.public_announcements: List[Dict[str, Any]] = []
        self.common_knowledge: Set[str] = set()

    def add_agent(self, agent_id: str) -> BeliefNetwork:
        """Add an agent to the network"""
        network = BeliefNetwork(agent_id)
        self.agents[agent_id] = network
        return network

    def public_announcement(self, proposition: str, source: str = "environment"):
        """Make a public announcement creating common knowledge"""
        announcement = {
            "proposition": proposition,
            "source": source,
            "timestamp": datetime.now(),
            "recipients": list(self.agents.keys())
        }
        self.public_announcements.append(announcement)

        # Add to all agents' beliefs and common knowledge
        for agent_id, agent in self.agents.items():
            belief = Belief(
                id=f"announcement_{len(self.public_announcements)}",
                proposition=proposition,
                belief_type=BeliefType.FACTUAL,
                credence=0.95,  # High credence for public announcements
                source=BeliefSource.TESTIMONY
            )
            agent.add_belief(belief)

        self.common_knowledge.add(proposition)

    def private_communication(self, sender: str, receiver: str,
                             proposition: str, credibility: float = 0.7):
        """Private communication between agents"""
        if receiver in self.agents:
            belief = Belief(
                id=f"private_{sender}_{len(self.agents[receiver].state.beliefs)}",
                proposition=proposition,
                belief_type=BeliefType.FACTUAL,
                credence=credibility,
                source=BeliefSource.TESTIMONY
            )
            self.agents[receiver].add_belief(belief)

            # Update beliefs about sender's beliefs
            self.agents[receiver].state.beliefs_about_others[sender] = {
                proposition: 0.9  # Assume sender believes what they say
            }

    def query_mutual_knowledge(self, agents: List[str], proposition: str) -> bool:
        """Check if proposition is mutual knowledge among agents"""
        for agent_id in agents:
            if agent_id not in self.agents:
                return False
            if self.agents[agent_id].state.get_credence(proposition) < 0.7:
                return False
        return True

    def query_common_knowledge(self, proposition: str) -> bool:
        """Check if proposition is common knowledge"""
        return proposition in self.common_knowledge


# Convenience functions
def create_information_dynamics() -> InformationDynamics:
    """Create information dynamics simulator"""
    return InformationDynamics()


def create_opinion_dynamics(model: OpinionModel = OpinionModel.BOUNDED_CONFIDENCE) -> OpinionDynamicsSimulator:
    """Create opinion dynamics simulator"""
    return OpinionDynamicsSimulator(model)


def demo_information_dynamics():
    """Demonstrate information dynamics"""
    print("=== Information Dynamics Demo ===")

    # Create network
    dynamics = create_information_dynamics()

    # Add agents
    for i in range(20):
        agent = AgentBeliefState(
            agent_id=f"agent_{i}",
            gullibility=random.uniform(0.3, 0.7),
            skepticism=random.uniform(0.3, 0.7),
            conformity=random.uniform(0.3, 0.7),
            influence=random.uniform(0.3, 0.7),
            connections={f"agent_{j}" for j in range(20) if j != i and random.random() < 0.3}
        )
        dynamics.add_agent(agent)

    # Introduce information
    true_info = InformationItem(
        id="true_news",
        content="Important factual information",
        source="agent_0",
        veracity=0.9,
        virality=0.6,
        credibility=0.7
    )
    dynamics.introduce_information(true_info)

    # Run cascade
    result = dynamics.run_cascade("true_news", steps=30)
    print(f"\nTrue info cascade:")
    print(f"  Adoption rate: {result['final_adoption_rate']:.1%}")
    print(f"  Total adopters: {result['total_adopters']}")
    print(f"  Steps: {result['steps_taken']}")

    # Opinion dynamics
    print("\n=== Opinion Dynamics Demo ===")
    opinions = create_opinion_dynamics()

    for i in range(10):
        opinions.add_agent(
            f"agent_{i}",
            {"climate": Opinion("climate", random.uniform(0, 1))},
            {f"agent_{j}" for j in range(10) if j != i and random.random() < 0.4}
        )

    # Run simulation
    for _ in range(20):
        opinions.step("climate")

    polarization = opinions.detect_polarization("climate")
    chambers = opinions.detect_echo_chambers()
    print(f"Polarization: {polarization:.2f}")
    print(f"Echo chambers: {len(chambers)}")

    return dynamics, opinions


if __name__ == "__main__":
    demo_information_dynamics()
