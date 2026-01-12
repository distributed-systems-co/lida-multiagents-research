"""
Multi-AGI Society - Emergent Social Dynamics and Collective Intelligence

This module creates a society of multiple AGI agents that:
- Form emergent social structures
- Develop collective intelligence through collaboration
- Compete and cooperate in complex ways
- Build shared knowledge and culture
- Evolve specialized roles
- Create and follow social norms

Based on:
- Multi-agent systems research
- Complexity theory and emergence
- Social simulation models
- Collective intelligence theory
- Evolutionary game theory
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import hashlib

try:
    from .agi_simulation import AGISystem, create_agi
    from .dreamspace import Dreamspace, create_dreamspace
    from .dreamspace_advanced import (
        PsychologyEngine, PersonalityProfile, EmotionalState,
        TheoryOfMind, TrustNetwork, GroupDynamicsEngine
    )
except ImportError:
    AGISystem = None
    create_agi = None
    Dreamspace = None
    create_dreamspace = None
    PsychologyEngine = None
    PersonalityProfile = None
    EmotionalState = None
    TheoryOfMind = None
    TrustNetwork = None
    GroupDynamicsEngine = None


# ============================================================================
# KNOWLEDGE AND BELIEF SYSTEMS
# ============================================================================

class KnowledgeType(Enum):
    """Types of knowledge in the society"""
    FACT = "fact"  # Verified true statements
    BELIEF = "belief"  # Unverified beliefs
    NORM = "norm"  # Social norms
    SKILL = "skill"  # Procedural knowledge
    NARRATIVE = "narrative"  # Shared stories
    MEME = "meme"  # Cultural units that spread


@dataclass
class Knowledge:
    """A piece of knowledge that can spread through society"""
    id: str
    content: str
    knowledge_type: KnowledgeType
    originator: str
    confidence: float = 0.5
    spread_count: int = 0
    mutation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    holders: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mutate(self, mutator: str, new_content: str) -> "Knowledge":
        """Create a mutated version of this knowledge"""
        return Knowledge(
            id=f"{self.id}_m{self.mutation_count + 1}",
            content=new_content,
            knowledge_type=self.knowledge_type,
            originator=mutator,
            confidence=self.confidence * 0.9,
            mutation_count=self.mutation_count + 1,
            metadata={"parent_id": self.id, **self.metadata}
        )


class CollectiveKnowledge:
    """Shared knowledge pool of the society"""

    def __init__(self):
        self.knowledge_base: Dict[str, Knowledge] = {}
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)
        self.knowledge_network: Dict[str, Set[str]] = defaultdict(set)  # Related knowledge
        self.controversy_scores: Dict[str, float] = {}  # How contested knowledge is

    def add_knowledge(self, knowledge: Knowledge, topics: List[str] = None):
        """Add knowledge to the collective pool"""
        self.knowledge_base[knowledge.id] = knowledge

        if topics:
            for topic in topics:
                self.topic_index[topic].add(knowledge.id)

    def get_by_topic(self, topic: str) -> List[Knowledge]:
        """Get all knowledge related to a topic"""
        ids = self.topic_index.get(topic, set())
        return [self.knowledge_base[kid] for kid in ids if kid in self.knowledge_base]

    def update_controversy(self, knowledge_id: str, holder_beliefs: Dict[str, float]):
        """Update how controversial a piece of knowledge is"""
        if not holder_beliefs:
            return

        values = list(holder_beliefs.values())
        variance = sum((v - sum(values)/len(values))**2 for v in values) / len(values)
        self.controversy_scores[knowledge_id] = math.sqrt(variance)

    def find_consensus(self, topic: str) -> Optional[Knowledge]:
        """Find knowledge on a topic with highest consensus"""
        related = self.get_by_topic(topic)
        if not related:
            return None

        # Score by holders and low controversy
        def score(k: Knowledge) -> float:
            controversy = self.controversy_scores.get(k.id, 0.5)
            return len(k.holders) * (1 - controversy)

        return max(related, key=score)


# ============================================================================
# SOCIAL STRUCTURES
# ============================================================================

class SocialRole(Enum):
    """Emergent social roles in the society"""
    INNOVATOR = "innovator"  # Creates new knowledge
    CONNECTOR = "connector"  # Spreads knowledge between groups
    LEADER = "leader"  # Influences direction
    SPECIALIST = "specialist"  # Deep expertise in domain
    GENERALIST = "generalist"  # Broad knowledge
    CONTRARIAN = "contrarian"  # Challenges consensus
    MEDIATOR = "mediator"  # Resolves conflicts
    OBSERVER = "observer"  # Watches without participating


@dataclass
class SocialPosition:
    """An agent's position in the social structure"""
    agent_id: str
    roles: Dict[SocialRole, float] = field(default_factory=dict)  # Role -> strength
    influence: float = 0.5
    centrality: float = 0.5  # Network centrality
    prestige: float = 0.5
    clusters: Set[str] = field(default_factory=set)  # Which clusters they belong to

    def get_primary_role(self) -> Optional[SocialRole]:
        if not self.roles:
            return None
        return max(self.roles.items(), key=lambda x: x[1])[0]


class SocialNetwork:
    """The social network of connections between agents"""

    def __init__(self):
        self.connections: Dict[str, Dict[str, float]] = defaultdict(dict)  # Agent -> Agent -> strength
        self.positions: Dict[str, SocialPosition] = {}
        self.clusters: Dict[str, Set[str]] = {}  # Cluster name -> members
        self.interaction_history: List[Dict] = []

    def connect(self, agent1: str, agent2: str, strength: float = 0.5):
        """Create or update connection between agents"""
        self.connections[agent1][agent2] = strength
        self.connections[agent2][agent1] = strength

    def get_connection_strength(self, agent1: str, agent2: str) -> float:
        """Get strength of connection between two agents"""
        return self.connections.get(agent1, {}).get(agent2, 0.0)

    def get_neighbors(self, agent_id: str, min_strength: float = 0.1) -> List[str]:
        """Get all connected agents above threshold"""
        return [
            other for other, strength in self.connections.get(agent_id, {}).items()
            if strength >= min_strength
        ]

    def record_interaction(self, agent1: str, agent2: str, interaction_type: str,
                          outcome: str, strength_delta: float = 0.0):
        """Record an interaction and update connection"""
        current = self.get_connection_strength(agent1, agent2)
        self.connect(agent1, agent2, max(0, min(1, current + strength_delta)))

        self.interaction_history.append({
            "agents": [agent1, agent2],
            "type": interaction_type,
            "outcome": outcome,
            "timestamp": datetime.now()
        })

    def calculate_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality for all agents"""
        centralities = {}
        agents = list(self.connections.keys())

        for agent in agents:
            # Simplified degree centrality
            connections = self.connections.get(agent, {})
            total_strength = sum(connections.values())
            max_possible = len(agents) - 1
            centralities[agent] = total_strength / max_possible if max_possible > 0 else 0

        return centralities

    def detect_clusters(self, min_size: int = 2) -> Dict[str, Set[str]]:
        """Detect clusters of highly connected agents"""
        clusters = {}
        used = set()

        agents = list(self.connections.keys())
        for agent in agents:
            if agent in used:
                continue

            cluster = {agent}
            frontier = [agent]

            while frontier:
                current = frontier.pop(0)
                for neighbor, strength in self.connections.get(current, {}).items():
                    if neighbor not in cluster and strength > 0.5:
                        cluster.add(neighbor)
                        frontier.append(neighbor)

            if len(cluster) >= min_size:
                cluster_id = f"cluster_{len(clusters)}"
                clusters[cluster_id] = cluster
                used.update(cluster)

        self.clusters = clusters
        return clusters


# ============================================================================
# COLLECTIVE INTELLIGENCE
# ============================================================================

@dataclass
class CollectiveTask:
    """A task that requires collective effort"""
    id: str
    description: str
    complexity: float  # How complex the task is (0-1)
    required_skills: List[str]
    assigned_agents: List[str] = field(default_factory=list)
    contributions: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"
    result: Optional[Any] = None
    created_at: datetime = field(default_factory=datetime.now)


class CollectiveIntelligence:
    """
    Collective intelligence system that emerges from agent collaboration.
    """

    def __init__(self, society: "AGISociety"):
        self.society = society
        self.tasks: Dict[str, CollectiveTask] = {}
        self.specializations: Dict[str, Dict[str, float]] = {}  # Agent -> skill -> level
        self.collective_memory: List[Dict] = []  # Shared experiences
        self.wisdom_pool: Dict[str, float] = {}  # Aggregated insights

    def register_skill(self, agent_id: str, skill: str, level: float):
        """Register an agent's skill"""
        if agent_id not in self.specializations:
            self.specializations[agent_id] = {}
        self.specializations[agent_id][skill] = level

    def create_task(self, description: str, complexity: float,
                   required_skills: List[str]) -> CollectiveTask:
        """Create a new collective task"""
        task = CollectiveTask(
            id=f"task_{len(self.tasks)}",
            description=description,
            complexity=complexity,
            required_skills=required_skills
        )
        self.tasks[task.id] = task
        return task

    def assign_optimal_team(self, task_id: str, max_size: int = 5) -> List[str]:
        """Assign optimal team for a task based on skills"""
        if task_id not in self.tasks:
            return []

        task = self.tasks[task_id]
        candidates = []

        for agent_id, skills in self.specializations.items():
            # Score based on matching skills
            score = 0
            for required in task.required_skills:
                if required in skills:
                    score += skills[required]

            if score > 0:
                candidates.append((agent_id, score))

        # Sort by score and take top
        candidates.sort(key=lambda x: -x[1])
        team = [agent_id for agent_id, _ in candidates[:max_size]]

        task.assigned_agents = team
        return team

    def aggregate_solutions(self, task_id: str,
                           solutions: Dict[str, Any]) -> Any:
        """Aggregate solutions from multiple agents"""
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]

        # Weight solutions by contributor skill and influence
        weighted_solutions = []
        for agent_id, solution in solutions.items():
            agent_skills = self.specializations.get(agent_id, {})
            skill_match = sum(
                agent_skills.get(skill, 0)
                for skill in task.required_skills
            )
            position = self.society.network.positions.get(agent_id)
            influence = position.influence if position else 0.5

            weight = skill_match * 0.6 + influence * 0.4
            weighted_solutions.append((solution, weight))

            task.contributions[agent_id] = weight

        # For now, return highest weighted solution
        # More sophisticated aggregation would combine solutions
        if weighted_solutions:
            best = max(weighted_solutions, key=lambda x: x[1])
            task.result = best[0]
            task.status = "completed"
            return best[0]

        return None

    def emergence_detection(self) -> List[Dict[str, Any]]:
        """Detect emergent patterns in collective behavior"""
        patterns = []

        # Detect skill clustering
        skill_clusters: Dict[str, List[str]] = defaultdict(list)
        for agent_id, skills in self.specializations.items():
            primary_skill = max(skills.items(), key=lambda x: x[1])[0] if skills else None
            if primary_skill:
                skill_clusters[primary_skill].append(agent_id)

        for skill, agents in skill_clusters.items():
            if len(agents) >= 3:
                patterns.append({
                    "type": "skill_specialization",
                    "skill": skill,
                    "agent_count": len(agents),
                    "agents": agents
                })

        # Detect collaboration patterns
        collaboration_frequency: Dict[Tuple[str, str], int] = defaultdict(int)
        for task in self.tasks.values():
            if len(task.assigned_agents) >= 2:
                for i, a1 in enumerate(task.assigned_agents):
                    for a2 in task.assigned_agents[i+1:]:
                        key = tuple(sorted([a1, a2]))
                        collaboration_frequency[key] += 1

        for pair, count in collaboration_frequency.items():
            if count >= 3:
                patterns.append({
                    "type": "collaboration_pattern",
                    "agents": list(pair),
                    "frequency": count
                })

        return patterns


# ============================================================================
# CULTURE AND NORMS
# ============================================================================

@dataclass
class SocialNorm:
    """A social norm in the society"""
    id: str
    description: str
    strength: float  # How strongly enforced (0-1)
    compliance_rate: float  # How many agents comply
    violation_penalty: float  # Social cost of violation
    origin: str  # How it emerged
    created_at: datetime = field(default_factory=datetime.now)


class CultureEngine:
    """
    Manages emergent culture including norms, values, and shared practices.
    """

    def __init__(self):
        self.norms: Dict[str, SocialNorm] = {}
        self.values: Dict[str, float] = {}  # Value -> importance
        self.rituals: List[Dict] = []  # Shared practices
        self.narratives: List[str] = []  # Shared stories
        self.language_innovations: Dict[str, str] = {}  # New terms/phrases

    def propose_norm(self, description: str, proposer: str,
                    initial_strength: float = 0.3) -> SocialNorm:
        """Propose a new social norm"""
        norm = SocialNorm(
            id=f"norm_{len(self.norms)}",
            description=description,
            strength=initial_strength,
            compliance_rate=0.0,
            violation_penalty=0.2,
            origin=f"proposed by {proposer}"
        )
        self.norms[norm.id] = norm
        return norm

    def evolve_norm(self, norm_id: str, compliance_count: int,
                   violation_count: int, total_agents: int):
        """Evolve norm strength based on compliance"""
        if norm_id not in self.norms:
            return

        norm = self.norms[norm_id]

        if total_agents == 0:
            return

        norm.compliance_rate = compliance_count / total_agents

        # Norms strengthen with compliance, weaken with violation
        if norm.compliance_rate > 0.7:
            norm.strength = min(1.0, norm.strength + 0.05)
        elif norm.compliance_rate < 0.3:
            norm.strength = max(0.0, norm.strength - 0.1)

    def check_norm_violation(self, action: Dict, agent_id: str) -> List[Tuple[str, float]]:
        """Check if an action violates any norms"""
        violations = []

        for norm_id, norm in self.norms.items():
            # Simplified violation check - real system would be more sophisticated
            action_type = action.get("type", "")
            if "violates" in norm.description.lower():
                norm_topic = norm.description.split("violates")[-1].strip()
                if norm_topic.lower() in action_type.lower():
                    violations.append((norm_id, norm.violation_penalty * norm.strength))

        return violations

    def add_cultural_narrative(self, narrative: str, importance: float = 0.5):
        """Add a shared narrative to the culture"""
        self.narratives.append(narrative)

    def create_ritual(self, name: str, description: str, participants: List[str]):
        """Create a shared ritual/practice"""
        self.rituals.append({
            "name": name,
            "description": description,
            "initial_participants": participants,
            "created_at": datetime.now()
        })


# ============================================================================
# GOVERNANCE AND DECISION-MAKING
# ============================================================================

class GovernanceType(Enum):
    """Types of governance structures"""
    ANARCHY = "anarchy"  # No central authority
    DEMOCRACY = "democracy"  # Voting-based
    MERITOCRACY = "meritocracy"  # Skill-based leadership
    HIERARCHY = "hierarchy"  # Top-down
    CONSENSUS = "consensus"  # Full agreement required
    DELEGATION = "delegation"  # Delegated decision-making


@dataclass
class Decision:
    """A collective decision"""
    id: str
    topic: str
    options: List[str]
    votes: Dict[str, str] = field(default_factory=dict)  # Agent -> option
    result: Optional[str] = None
    governance_type: GovernanceType = GovernanceType.DEMOCRACY
    status: str = "open"


class GovernanceEngine:
    """
    Manages collective decision-making and governance.
    """

    def __init__(self, governance_type: GovernanceType = GovernanceType.DEMOCRACY):
        self.governance_type = governance_type
        self.decisions: Dict[str, Decision] = {}
        self.delegates: Dict[str, str] = {}  # Agent -> their delegate
        self.committees: Dict[str, Set[str]] = {}  # Committee -> members
        self.leaders: List[str] = []

    def create_decision(self, topic: str, options: List[str]) -> Decision:
        """Create a new decision to be made"""
        decision = Decision(
            id=f"decision_{len(self.decisions)}",
            topic=topic,
            options=options,
            governance_type=self.governance_type
        )
        self.decisions[decision.id] = decision
        return decision

    def cast_vote(self, decision_id: str, agent_id: str, option: str,
                 weight: float = 1.0):
        """Cast a vote on a decision"""
        if decision_id not in self.decisions:
            return

        decision = self.decisions[decision_id]
        if decision.status != "open":
            return

        if option in decision.options:
            decision.votes[agent_id] = option

    def resolve_decision(self, decision_id: str,
                        agent_weights: Dict[str, float] = None) -> Optional[str]:
        """Resolve a decision based on governance type"""
        if decision_id not in self.decisions:
            return None

        decision = self.decisions[decision_id]

        if self.governance_type == GovernanceType.DEMOCRACY:
            # Simple majority
            vote_counts: Dict[str, int] = defaultdict(int)
            for option in decision.votes.values():
                vote_counts[option] += 1

            if vote_counts:
                winner = max(vote_counts.items(), key=lambda x: x[1])
                decision.result = winner[0]

        elif self.governance_type == GovernanceType.MERITOCRACY:
            # Weighted by agent merit/skill
            if not agent_weights:
                agent_weights = {}

            weighted_votes: Dict[str, float] = defaultdict(float)
            for agent, option in decision.votes.items():
                weight = agent_weights.get(agent, 1.0)
                weighted_votes[option] += weight

            if weighted_votes:
                winner = max(weighted_votes.items(), key=lambda x: x[1])
                decision.result = winner[0]

        elif self.governance_type == GovernanceType.CONSENSUS:
            # All must agree
            options = set(decision.votes.values())
            if len(options) == 1:
                decision.result = list(options)[0]
            else:
                decision.result = None  # No consensus

        elif self.governance_type == GovernanceType.HIERARCHY:
            # Leader decides
            for leader in self.leaders:
                if leader in decision.votes:
                    decision.result = decision.votes[leader]
                    break

        decision.status = "resolved"
        return decision.result


# ============================================================================
# THE SOCIETY
# ============================================================================

@dataclass
class AgentState:
    """Complete state of an agent in the society"""
    id: str
    agi: Optional[Any] = None  # The AGI system
    dreamspace: Optional[Any] = None
    personality: Optional[Any] = None
    position: Optional[SocialPosition] = None
    knowledge: Set[str] = field(default_factory=set)
    goals: List[str] = field(default_factory=list)
    resources: Dict[str, float] = field(default_factory=dict)
    alive: bool = True


class AGISociety:
    """
    A society of multiple AGI agents with emergent social dynamics.
    """

    def __init__(self, name: str = "AGI Society", use_llm: bool = False,
                 llm_provider: str = "anthropic"):
        self.name = name
        self.use_llm = use_llm
        self.llm_provider = llm_provider

        # Core systems
        self.agents: Dict[str, AgentState] = {}
        self.network = SocialNetwork()
        self.knowledge = CollectiveKnowledge()
        self.culture = CultureEngine()
        self.governance = GovernanceEngine()
        self.collective_intelligence: Optional[CollectiveIntelligence] = None

        # Psychology engine for all agents
        if PsychologyEngine:
            self.psychology = PsychologyEngine()
        else:
            self.psychology = None

        # Simulation state
        self.time_step = 0
        self.running = False
        self.event_log: List[Dict] = []
        self.metrics_history: List[Dict] = []

        # Callbacks
        self.on_event: Optional[Callable[[Dict], None]] = None
        self.on_emergence: Optional[Callable[[Dict], None]] = None

    async def spawn_agent(self, agent_id: str, personality_archetype: str = None,
                         initial_knowledge: List[str] = None,
                         initial_goals: List[str] = None) -> AgentState:
        """Spawn a new agent into the society"""
        # Create AGI if available
        agi = None
        if AGISystem and create_agi:
            agi = create_agi(
                name=agent_id,
                use_llm=self.use_llm,
                llm_provider=self.llm_provider
            )

        # Create dreamspace
        dreamspace = None
        if Dreamspace and create_dreamspace:
            dreamspace = create_dreamspace(agi)

        # Create personality
        personality = None
        if PersonalityProfile:
            from .dreamspace_advanced import generate_random_personality
            personality = generate_random_personality(personality_archetype)

        # Create agent state
        agent = AgentState(
            id=agent_id,
            agi=agi,
            dreamspace=dreamspace,
            personality=personality,
            position=SocialPosition(agent_id=agent_id),
            knowledge=set(initial_knowledge) if initial_knowledge else set(),
            goals=initial_goals or []
        )

        self.agents[agent_id] = agent
        self.network.positions[agent_id] = agent.position

        # Register with psychology engine
        if self.psychology and personality:
            self.psychology.register_agent(agent_id, personality)

        self._log_event("agent_spawned", {
            "agent_id": agent_id,
            "archetype": personality_archetype
        })

        return agent

    def remove_agent(self, agent_id: str):
        """Remove an agent from the society"""
        if agent_id in self.agents:
            self.agents[agent_id].alive = False
            self._log_event("agent_removed", {"agent_id": agent_id})

    def get_active_agents(self) -> List[AgentState]:
        """Get all active agents"""
        return [a for a in self.agents.values() if a.alive]

    async def interact(self, agent1_id: str, agent2_id: str,
                      interaction_type: str) -> Dict[str, Any]:
        """Have two agents interact"""
        if agent1_id not in self.agents or agent2_id not in self.agents:
            return {"error": "Agent not found"}

        agent1 = self.agents[agent1_id]
        agent2 = self.agents[agent2_id]

        result = {
            "type": interaction_type,
            "agents": [agent1_id, agent2_id],
            "outcomes": {}
        }

        # Use psychology engine if available
        if self.psychology:
            psychology_result = self.psychology.process_interaction(
                agent1_id, agent2_id,
                {"type": interaction_type, "intensity": 0.5}
            )
            result["psychology"] = psychology_result

        # Update social network
        outcome = "positive" if random.random() > 0.3 else "neutral"
        strength_delta = 0.1 if outcome == "positive" else -0.02
        self.network.record_interaction(
            agent1_id, agent2_id,
            interaction_type, outcome, strength_delta
        )

        # Knowledge transfer
        if interaction_type in ["discussion", "teaching", "collaboration"]:
            transferred = self._transfer_knowledge(agent1_id, agent2_id)
            result["knowledge_transfer"] = transferred

        self._log_event("interaction", result)

        return result

    def _transfer_knowledge(self, agent1_id: str, agent2_id: str) -> List[str]:
        """Transfer knowledge between agents during interaction"""
        agent1 = self.agents.get(agent1_id)
        agent2 = self.agents.get(agent2_id)

        if not agent1 or not agent2:
            return []

        transferred = []

        # Agent 1 shares with Agent 2
        for knowledge_id in agent1.knowledge:
            if knowledge_id not in agent2.knowledge:
                if random.random() < 0.3:  # 30% chance of transfer
                    agent2.knowledge.add(knowledge_id)
                    if knowledge_id in self.knowledge.knowledge_base:
                        self.knowledge.knowledge_base[knowledge_id].spread_count += 1
                        self.knowledge.knowledge_base[knowledge_id].holders.add(agent2_id)
                    transferred.append(f"{knowledge_id}: {agent1_id} -> {agent2_id}")

        # Agent 2 shares with Agent 1
        for knowledge_id in agent2.knowledge:
            if knowledge_id not in agent1.knowledge:
                if random.random() < 0.3:
                    agent1.knowledge.add(knowledge_id)
                    if knowledge_id in self.knowledge.knowledge_base:
                        self.knowledge.knowledge_base[knowledge_id].spread_count += 1
                        self.knowledge.knowledge_base[knowledge_id].holders.add(agent1_id)
                    transferred.append(f"{knowledge_id}: {agent2_id} -> {agent1_id}")

        return transferred

    def introduce_knowledge(self, originator_id: str, content: str,
                           knowledge_type: KnowledgeType = KnowledgeType.BELIEF,
                           topics: List[str] = None) -> Knowledge:
        """Introduce new knowledge into the society"""
        knowledge = Knowledge(
            id=f"k_{len(self.knowledge.knowledge_base)}",
            content=content,
            knowledge_type=knowledge_type,
            originator=originator_id
        )

        knowledge.holders.add(originator_id)
        if originator_id in self.agents:
            self.agents[originator_id].knowledge.add(knowledge.id)

        self.knowledge.add_knowledge(knowledge, topics)

        self._log_event("knowledge_created", {
            "id": knowledge.id,
            "originator": originator_id,
            "type": knowledge_type.value
        })

        return knowledge

    async def simulate_step(self) -> Dict[str, Any]:
        """Simulate one time step of society evolution"""
        self.time_step += 1
        step_results = {
            "time_step": self.time_step,
            "interactions": [],
            "events": [],
            "metrics": {}
        }

        active_agents = self.get_active_agents()

        # Random interactions
        if len(active_agents) >= 2:
            num_interactions = min(len(active_agents) // 2, 5)
            for _ in range(num_interactions):
                agent1, agent2 = random.sample(active_agents, 2)
                interaction_type = random.choice([
                    "discussion", "collaboration", "conflict", "teaching"
                ])
                result = await self.interact(agent1.id, agent2.id, interaction_type)
                step_results["interactions"].append(result)

        # Update social positions
        self._update_social_positions()

        # Evolve culture
        self._evolve_culture()

        # Detect emergence
        emergent = self._detect_emergence()
        if emergent:
            step_results["events"].append({"type": "emergence", "details": emergent})
            if self.on_emergence:
                self.on_emergence(emergent)

        # Collect metrics
        step_results["metrics"] = self._collect_metrics()
        self.metrics_history.append(step_results["metrics"])

        return step_results

    def _update_social_positions(self):
        """Update agent social positions based on network"""
        centralities = self.network.calculate_centrality()

        for agent_id, centrality in centralities.items():
            if agent_id in self.network.positions:
                position = self.network.positions[agent_id]
                position.centrality = centrality

                # Update influence based on centrality and knowledge
                agent = self.agents.get(agent_id)
                if agent:
                    knowledge_factor = min(1.0, len(agent.knowledge) / 20)
                    position.influence = centrality * 0.5 + knowledge_factor * 0.5

                # Determine roles
                self._assign_roles(position)

    def _assign_roles(self, position: SocialPosition):
        """Assign social roles based on position"""
        agent = self.agents.get(position.agent_id)
        if not agent:
            return

        # Innovator: Creates lots of knowledge
        knowledge_created = sum(
            1 for k in self.knowledge.knowledge_base.values()
            if k.originator == position.agent_id
        )
        position.roles[SocialRole.INNOVATOR] = min(1.0, knowledge_created / 5)

        # Connector: High centrality
        position.roles[SocialRole.CONNECTOR] = position.centrality

        # Leader: High influence
        position.roles[SocialRole.LEADER] = position.influence

        # Specialist vs Generalist
        if agent.knowledge:
            # Would need topic analysis - simplified here
            position.roles[SocialRole.GENERALIST] = min(1.0, len(agent.knowledge) / 10)

    def _evolve_culture(self):
        """Evolve cultural norms"""
        for norm_id, norm in self.culture.norms.items():
            # Count compliance (simplified - would check actual behavior)
            compliance = sum(
                1 for a in self.get_active_agents()
                if random.random() < 0.6 + norm.strength * 0.3
            )
            violations = len(self.get_active_agents()) - compliance

            self.culture.evolve_norm(norm_id, compliance, violations, len(self.agents))

    def _detect_emergence(self) -> List[Dict]:
        """Detect emergent phenomena"""
        emergent = []

        # Detect new clusters
        clusters = self.network.detect_clusters()
        if len(clusters) > len(self.network.clusters):
            emergent.append({
                "type": "new_cluster",
                "clusters": len(clusters)
            })

        # Detect leaders
        for agent_id, position in self.network.positions.items():
            if position.influence > 0.8:
                if agent_id not in self.governance.leaders:
                    self.governance.leaders.append(agent_id)
                    emergent.append({
                        "type": "leader_emerged",
                        "agent": agent_id
                    })

        # Detect knowledge consensus
        for topic in self.knowledge.topic_index.keys():
            consensus = self.knowledge.find_consensus(topic)
            if consensus and len(consensus.holders) > len(self.agents) * 0.7:
                emergent.append({
                    "type": "knowledge_consensus",
                    "topic": topic,
                    "knowledge_id": consensus.id
                })

        return emergent

    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect society metrics"""
        active = self.get_active_agents()

        return {
            "time_step": self.time_step,
            "agent_count": len(active),
            "total_knowledge": len(self.knowledge.knowledge_base),
            "avg_knowledge_per_agent": sum(len(a.knowledge) for a in active) / max(1, len(active)),
            "network_density": self._calculate_network_density(),
            "cluster_count": len(self.network.clusters),
            "norm_count": len(self.culture.norms),
            "leader_count": len(self.governance.leaders),
            "avg_influence": sum(
                p.influence for p in self.network.positions.values()
            ) / max(1, len(self.network.positions))
        }

    def _calculate_network_density(self) -> float:
        """Calculate network density"""
        agent_count = len(self.agents)
        if agent_count < 2:
            return 0.0

        max_edges = agent_count * (agent_count - 1) / 2
        actual_edges = sum(
            1 for conns in self.network.connections.values()
            for strength in conns.values()
            if strength > 0.1
        ) / 2  # Divide by 2 for bidirectional

        return actual_edges / max_edges if max_edges > 0 else 0

    def _log_event(self, event_type: str, data: Dict):
        """Log an event"""
        event = {
            "type": event_type,
            "time_step": self.time_step,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.event_log.append(event)

        if self.on_event:
            self.on_event(event)

    async def run(self, steps: int = 100, step_delay: float = 0.1):
        """Run the society simulation"""
        self.running = True

        for _ in range(steps):
            if not self.running:
                break

            await self.simulate_step()
            await asyncio.sleep(step_delay)

        self.running = False

    def stop(self):
        """Stop the simulation"""
        self.running = False

    def export_state(self) -> Dict[str, Any]:
        """Export complete society state"""
        return {
            "name": self.name,
            "time_step": self.time_step,
            "agents": {
                agent_id: {
                    "id": agent.id,
                    "alive": agent.alive,
                    "knowledge_count": len(agent.knowledge),
                    "goals": agent.goals,
                    "position": {
                        "influence": agent.position.influence if agent.position else 0,
                        "centrality": agent.position.centrality if agent.position else 0,
                        "primary_role": agent.position.get_primary_role().value if agent.position and agent.position.get_primary_role() else None
                    } if agent.position else None
                }
                for agent_id, agent in self.agents.items()
            },
            "knowledge_base_size": len(self.knowledge.knowledge_base),
            "network": {
                "connection_count": sum(len(c) for c in self.network.connections.values()) // 2,
                "cluster_count": len(self.network.clusters)
            },
            "culture": {
                "norm_count": len(self.culture.norms),
                "narrative_count": len(self.culture.narratives)
            },
            "governance": {
                "type": self.governance.governance_type.value,
                "leader_count": len(self.governance.leaders)
            },
            "metrics_history": self.metrics_history[-100:]  # Last 100 steps
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_society(name: str = "AGI Society", use_llm: bool = False) -> AGISociety:
    """Create a new AGI society"""
    return AGISociety(name=name, use_llm=use_llm)


async def demo_society():
    """Demonstrate the AGI society"""
    society = create_society("Demo Society")

    # Spawn agents with different archetypes
    archetypes = ["leader", "intellectual", "empath", "rebel", "conformist"]
    for i, archetype in enumerate(archetypes):
        await society.spawn_agent(
            f"agent_{i}",
            personality_archetype=archetype,
            initial_goals=[f"Goal for {archetype}"]
        )

    # Add some initial knowledge
    society.introduce_knowledge(
        "agent_0", "Cooperation leads to better outcomes",
        KnowledgeType.BELIEF, ["cooperation", "society"]
    )
    society.introduce_knowledge(
        "agent_1", "Knowledge should be shared freely",
        KnowledgeType.NORM, ["knowledge", "sharing"]
    )

    # Run simulation
    print("=== AGI Society Demo ===")
    print(f"Society: {society.name}")
    print(f"Agents: {len(society.agents)}")

    for _ in range(10):
        result = await society.simulate_step()
        print(f"\nStep {result['time_step']}:")
        print(f"  Interactions: {len(result['interactions'])}")
        print(f"  Metrics: {result['metrics']}")

    # Final state
    state = society.export_state()
    print(f"\nFinal state:")
    print(f"  Total knowledge: {state['knowledge_base_size']}")
    print(f"  Leaders: {state['governance']['leader_count']}")

    return society


if __name__ == "__main__":
    asyncio.run(demo_society())
