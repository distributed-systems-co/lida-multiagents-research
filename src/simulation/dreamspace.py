"""
Dreamspace Module - Subconscious Processing for AGI Systems

Implements advanced cognitive processes that occur "below" conscious awareness:
- Memory consolidation and integration
- Creative recombination of concepts
- Pattern discovery across experiences
- Emotional processing and regulation
- Predictive world modeling
- Counterfactual simulation
- Insight generation

Based on theories of:
- Default Mode Network (DMN) activity
- REM sleep memory consolidation
- Incubation effects in creativity
- Predictive processing / Active inference
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import hashlib


class DreamState(Enum):
    """States of the dreamspace processing"""
    IDLE = "idle"
    CONSOLIDATING = "consolidating"  # Memory consolidation
    WANDERING = "wandering"  # Mind-wandering / DMN activity
    INCUBATING = "incubating"  # Problem incubation
    SIMULATING = "simulating"  # Counterfactual simulation
    INTEGRATING = "integrating"  # Cross-modal integration
    CREATING = "creating"  # Creative synthesis
    PREDICTING = "predicting"  # Future scenario modeling


class InsightType(Enum):
    """Types of insights that can emerge"""
    CONNECTION = "connection"  # Linking disparate concepts
    PATTERN = "pattern"  # Discovering hidden patterns
    ANALOGY = "analogy"  # Finding structural similarities
    CONTRADICTION = "contradiction"  # Detecting inconsistencies
    PREDICTION = "prediction"  # Anticipating outcomes
    SOLUTION = "solution"  # Problem solutions
    REFRAME = "reframe"  # New perspectives
    SYNTHESIS = "synthesis"  # Combining ideas


@dataclass
class ConceptNode:
    """A node in the semantic/conceptual network"""
    id: str
    content: str
    concept_type: str  # fact, belief, memory, emotion, goal, etc.
    activation: float = 0.0
    valence: float = 0.0  # Emotional valence (-1 to 1)
    arousal: float = 0.0  # Emotional arousal (0 to 1)
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    connections: Dict[str, float] = field(default_factory=dict)  # id -> weight
    metadata: Dict[str, Any] = field(default_factory=dict)

    def decay(self, rate: float = 0.01):
        """Apply activation decay"""
        self.activation *= (1 - rate)

    def activate(self, amount: float):
        """Increase activation with ceiling"""
        self.activation = min(1.0, self.activation + amount)
        self.access_count += 1


@dataclass
class DreamFragment:
    """A fragment of dream/subconscious content"""
    id: str
    content: str
    source_concepts: List[str]
    dream_state: DreamState
    coherence: float  # How coherent/logical (0-1)
    novelty: float  # How novel/creative (0-1)
    emotional_charge: float  # Emotional intensity
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Insight:
    """An insight that emerges from subconscious processing"""
    id: str
    insight_type: InsightType
    content: str
    confidence: float
    source_concepts: List[str]
    trigger: str  # What triggered this insight
    implications: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.insight_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "sources": self.source_concepts,
            "trigger": self.trigger,
            "implications": self.implications,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class PredictiveModel:
    """A predictive model about some aspect of the world"""
    id: str
    domain: str  # What domain this model covers
    hypothesis: str
    evidence_for: List[str]
    evidence_against: List[str]
    confidence: float
    predictions: List[Dict[str, Any]]
    last_updated: datetime = field(default_factory=datetime.now)
    accuracy_history: List[float] = field(default_factory=list)


class SemanticNetwork:
    """
    A spreading-activation semantic network for concept representation.
    Enables associative retrieval and pattern discovery.
    """

    def __init__(self):
        self.nodes: Dict[str, ConceptNode] = {}
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        self.activation_history: List[Dict[str, float]] = []

    def add_node(self, node: ConceptNode):
        """Add a concept node to the network"""
        self.nodes[node.id] = node
        self.type_index[node.concept_type].add(node.id)

    def connect(self, id1: str, id2: str, weight: float = 0.5, bidirectional: bool = True):
        """Connect two nodes"""
        if id1 in self.nodes and id2 in self.nodes:
            self.nodes[id1].connections[id2] = weight
            if bidirectional:
                self.nodes[id2].connections[id1] = weight

    def spread_activation(self, source_ids: List[str], initial_activation: float = 1.0,
                         decay: float = 0.3, threshold: float = 0.1, max_steps: int = 5):
        """
        Spread activation through the network from source nodes.
        Returns activated nodes above threshold.
        """
        # Initialize activation
        for node in self.nodes.values():
            node.activation = 0.0

        for source_id in source_ids:
            if source_id in self.nodes:
                self.nodes[source_id].activate(initial_activation)

        activated = set(source_ids)
        frontier = list(source_ids)

        for step in range(max_steps):
            new_frontier = []
            for node_id in frontier:
                node = self.nodes.get(node_id)
                if not node:
                    continue

                # Spread to connected nodes
                for connected_id, weight in node.connections.items():
                    if connected_id in self.nodes:
                        spread_amount = node.activation * weight * (1 - decay)
                        self.nodes[connected_id].activate(spread_amount)

                        if self.nodes[connected_id].activation >= threshold:
                            if connected_id not in activated:
                                activated.add(connected_id)
                                new_frontier.append(connected_id)

            frontier = new_frontier
            if not frontier:
                break

        # Record activation state
        self.activation_history.append({
            node_id: node.activation
            for node_id, node in self.nodes.items()
            if node.activation > 0
        })

        return [(node_id, self.nodes[node_id].activation)
                for node_id in activated
                if self.nodes[node_id].activation >= threshold]

    def find_bridges(self, threshold: float = 0.3) -> List[Tuple[str, str, List[str]]]:
        """
        Find concepts that bridge otherwise disconnected clusters.
        Returns: [(concept1, concept2, bridge_concepts), ...]
        """
        bridges = []

        # Find nodes with diverse connections
        for node_id, node in self.nodes.items():
            if len(node.connections) < 3:
                continue

            connected_types = set()
            for connected_id in node.connections:
                if connected_id in self.nodes:
                    connected_types.add(self.nodes[connected_id].concept_type)

            if len(connected_types) >= 3:
                # This is a bridge node
                for c1 in node.connections:
                    for c2 in node.connections:
                        if c1 < c2:  # Avoid duplicates
                            bridges.append((c1, c2, [node_id]))

        return bridges

    def cluster_by_activation(self, min_size: int = 2) -> List[Set[str]]:
        """Find clusters of co-activated nodes"""
        if not self.activation_history:
            return []

        recent = self.activation_history[-10:]  # Last 10 activation patterns

        # Find frequently co-activated pairs
        coactivation = defaultdict(int)
        for pattern in recent:
            active = [k for k, v in pattern.items() if v > 0.3]
            for i, n1 in enumerate(active):
                for n2 in active[i+1:]:
                    pair = tuple(sorted([n1, n2]))
                    coactivation[pair] += 1

        # Build clusters from frequent pairs
        clusters = []
        used = set()

        for (n1, n2), count in sorted(coactivation.items(), key=lambda x: -x[1]):
            if count < 2:
                continue
            if n1 in used or n2 in used:
                continue

            cluster = {n1, n2}
            # Expand cluster
            for (a, b), c in coactivation.items():
                if c >= count - 1:
                    if a in cluster or b in cluster:
                        cluster.add(a)
                        cluster.add(b)

            if len(cluster) >= min_size:
                clusters.append(cluster)
                used.update(cluster)

        return clusters


class Dreamspace:
    """
    The subconscious processing system for an AGI.
    Runs background processes that consolidate memory,
    discover patterns, incubate problems, and generate insights.
    """

    def __init__(self, agi_system: Optional[Any] = None):
        self.agi = agi_system
        self.state = DreamState.IDLE
        self.semantic_network = SemanticNetwork()
        self.dream_fragments: List[DreamFragment] = []
        self.insights: List[Insight] = []
        self.predictive_models: Dict[str, PredictiveModel] = {}
        self.incubating_problems: List[Dict[str, Any]] = []
        self.emotional_buffer: List[Dict[str, Any]] = []
        self.running = False
        self._cycle_count = 0
        self._last_consolidation = datetime.now()

        # Configuration
        self.consolidation_interval = timedelta(minutes=5)
        self.wandering_probability = 0.3
        self.creativity_temperature = 0.7
        self.insight_threshold = 0.6

        # Callbacks for integration with main AGI
        self.on_insight: Optional[Callable[[Insight], None]] = None
        self.on_prediction: Optional[Callable[[PredictiveModel], None]] = None

    async def start(self):
        """Start the dreamspace background processing"""
        self.running = True
        asyncio.create_task(self._background_loop())

    async def stop(self):
        """Stop background processing"""
        self.running = False

    async def _background_loop(self):
        """Main background processing loop"""
        while self.running:
            self._cycle_count += 1

            # Determine what to do this cycle
            if datetime.now() - self._last_consolidation > self.consolidation_interval:
                await self._consolidate_memories()
                self._last_consolidation = datetime.now()
            elif self.incubating_problems and random.random() < 0.4:
                await self._incubate_problem()
            elif random.random() < self.wandering_probability:
                await self._mind_wander()
            else:
                await self._passive_integration()

            # Always try to generate insights
            await self._attempt_insight_generation()

            # Apply decay
            for node in self.semantic_network.nodes.values():
                node.decay(0.005)

            await asyncio.sleep(0.1)  # Prevent tight loop

    def ingest_experience(self, content: str, concept_type: str = "experience",
                         valence: float = 0.0, arousal: float = 0.0,
                         related_concepts: List[str] = None):
        """
        Ingest a new experience into the dreamspace.
        This will be processed subconsciously.
        """
        node_id = hashlib.md5(f"{content}{datetime.now()}".encode()).hexdigest()[:12]

        node = ConceptNode(
            id=node_id,
            content=content,
            concept_type=concept_type,
            activation=0.8,  # New experiences start highly activated
            valence=valence,
            arousal=arousal
        )

        self.semantic_network.add_node(node)

        # Connect to related concepts
        if related_concepts:
            for related_id in related_concepts:
                if related_id in self.semantic_network.nodes:
                    # Weight based on semantic similarity (simplified)
                    weight = 0.5 + random.uniform(-0.2, 0.2)
                    self.semantic_network.connect(node_id, related_id, weight)

        # Find and connect to similar existing nodes
        self._auto_connect_similar(node)

        # If emotionally significant, add to emotional buffer
        if abs(valence) > 0.5 or arousal > 0.6:
            self.emotional_buffer.append({
                "node_id": node_id,
                "valence": valence,
                "arousal": arousal,
                "content": content,
                "timestamp": datetime.now()
            })

        return node_id

    def _auto_connect_similar(self, new_node: ConceptNode, top_k: int = 5):
        """Automatically connect to similar nodes based on content overlap"""
        new_words = set(new_node.content.lower().split())

        similarities = []
        for node_id, node in self.semantic_network.nodes.items():
            if node_id == new_node.id:
                continue
            existing_words = set(node.content.lower().split())
            overlap = len(new_words & existing_words)
            if overlap > 0:
                similarity = overlap / max(len(new_words), len(existing_words))
                similarities.append((node_id, similarity))

        # Connect to top-k similar
        for node_id, sim in sorted(similarities, key=lambda x: -x[1])[:top_k]:
            self.semantic_network.connect(new_node.id, node_id, sim)

    def submit_problem(self, problem: str, context: Dict[str, Any] = None,
                      priority: float = 0.5):
        """Submit a problem for subconscious incubation"""
        self.incubating_problems.append({
            "problem": problem,
            "context": context or {},
            "priority": priority,
            "submitted_at": datetime.now(),
            "attempts": 0,
            "partial_solutions": []
        })

    async def _consolidate_memories(self):
        """
        Memory consolidation process - like what happens during sleep.
        Strengthens important connections, prunes weak ones,
        and integrates new memories with old.
        """
        self.state = DreamState.CONSOLIDATING

        # Find highly activated nodes
        active_nodes = [
            (node_id, node)
            for node_id, node in self.semantic_network.nodes.items()
            if node.activation > 0.3 or node.access_count > 5
        ]

        for node_id, node in active_nodes:
            # Strengthen connections to frequently co-activated nodes
            clusters = self.semantic_network.cluster_by_activation()
            for cluster in clusters:
                if node_id in cluster:
                    for other_id in cluster:
                        if other_id != node_id:
                            current = node.connections.get(other_id, 0)
                            node.connections[other_id] = min(1.0, current + 0.1)

            # Prune weak connections
            weak_connections = [
                conn_id for conn_id, weight in node.connections.items()
                if weight < 0.1
            ]
            for conn_id in weak_connections:
                del node.connections[conn_id]

        # Generate consolidation dream fragment
        if active_nodes:
            sample_nodes = random.sample(active_nodes, min(3, len(active_nodes)))
            fragment = DreamFragment(
                id=f"dream_{self._cycle_count}",
                content=f"Consolidating: {' + '.join(n.content[:50] for _, n in sample_nodes)}",
                source_concepts=[node_id for node_id, _ in sample_nodes],
                dream_state=DreamState.CONSOLIDATING,
                coherence=0.7,
                novelty=0.2,
                emotional_charge=sum(n.arousal for _, n in sample_nodes) / len(sample_nodes)
            )
            self.dream_fragments.append(fragment)

    async def _mind_wander(self):
        """
        Mind-wandering / Default Mode Network activity.
        Random exploration of the semantic network.
        """
        self.state = DreamState.WANDERING

        if not self.semantic_network.nodes:
            return

        # Start from random node
        start_node = random.choice(list(self.semantic_network.nodes.keys()))

        # Spread activation with high creativity (low decay)
        activated = self.semantic_network.spread_activation(
            [start_node],
            initial_activation=0.8,
            decay=0.2,
            threshold=0.1,
            max_steps=7
        )

        # Look for interesting combinations
        if len(activated) >= 3:
            # Find bridge concepts
            bridges = self.semantic_network.find_bridges()
            if bridges:
                # This could lead to an insight
                bridge = random.choice(bridges)
                await self._evaluate_bridge(bridge)

    async def _evaluate_bridge(self, bridge: Tuple[str, str, List[str]]):
        """Evaluate if a bridge between concepts leads to an insight"""
        concept1_id, concept2_id, bridge_ids = bridge

        if concept1_id not in self.semantic_network.nodes:
            return
        if concept2_id not in self.semantic_network.nodes:
            return

        concept1 = self.semantic_network.nodes[concept1_id]
        concept2 = self.semantic_network.nodes[concept2_id]

        # Check if these concepts are from different domains
        if concept1.concept_type != concept2.concept_type:
            # Potential cross-domain insight
            confidence = 0.3 + (len(bridge_ids) * 0.1)

            if confidence > self.insight_threshold:
                insight = Insight(
                    id=f"insight_{len(self.insights)}",
                    insight_type=InsightType.CONNECTION,
                    content=f"Connection between '{concept1.content[:50]}' and '{concept2.content[:50]}'",
                    confidence=confidence,
                    source_concepts=[concept1_id, concept2_id] + bridge_ids,
                    trigger="mind_wandering",
                    implications=[
                        f"These concepts share structural similarity via {len(bridge_ids)} bridge concepts"
                    ]
                )
                self._register_insight(insight)

    async def _incubate_problem(self):
        """
        Work on an incubating problem subconsciously.
        This is the "aha moment" generator.
        """
        self.state = DreamState.INCUBATING

        if not self.incubating_problems:
            return

        # Select problem to work on (priority + time waiting)
        problem = max(
            self.incubating_problems,
            key=lambda p: p["priority"] + (datetime.now() - p["submitted_at"]).seconds / 3600
        )

        problem["attempts"] += 1

        # Extract key concepts from problem
        problem_words = problem["problem"].lower().split()

        # Find related nodes in semantic network
        related_nodes = []
        for node_id, node in self.semantic_network.nodes.items():
            node_words = set(node.content.lower().split())
            if node_words & set(problem_words):
                related_nodes.append(node_id)

        if related_nodes:
            # Spread activation from problem-related nodes
            activated = self.semantic_network.spread_activation(
                related_nodes,
                initial_activation=1.0,
                decay=0.15,  # Low decay for creative exploration
                max_steps=10
            )

            # Look for unexpected activations
            unexpected = [
                (node_id, activation)
                for node_id, activation in activated
                if node_id not in related_nodes and activation > 0.4
            ]

            if unexpected:
                # Potential solution insight!
                best = max(unexpected, key=lambda x: x[1])
                node = self.semantic_network.nodes[best[0]]

                partial_solution = {
                    "concept": node.content,
                    "relevance": best[1],
                    "found_at": datetime.now()
                }
                problem["partial_solutions"].append(partial_solution)

                if best[1] > 0.7:
                    # Strong enough for insight
                    insight = Insight(
                        id=f"solution_{len(self.insights)}",
                        insight_type=InsightType.SOLUTION,
                        content=f"For problem '{problem['problem'][:50]}...': Consider '{node.content}'",
                        confidence=best[1],
                        source_concepts=related_nodes + [best[0]],
                        trigger="incubation",
                        implications=["Unexpected connection found through subconscious search"]
                    )
                    self._register_insight(insight)

    async def _passive_integration(self):
        """
        Passive integration of recent experiences.
        Looking for patterns and contradictions.
        """
        self.state = DreamState.INTEGRATING

        # Get recent high-activation nodes
        recent = sorted(
            self.semantic_network.nodes.values(),
            key=lambda n: n.activation,
            reverse=True
        )[:10]

        # Look for contradictions
        for i, node1 in enumerate(recent):
            for node2 in recent[i+1:]:
                # Check for opposing valence with similar content
                if (abs(node1.valence - node2.valence) > 1.0 and
                    self._content_overlap(node1.content, node2.content) > 0.3):

                    insight = Insight(
                        id=f"contradiction_{len(self.insights)}",
                        insight_type=InsightType.CONTRADICTION,
                        content=f"Tension between: '{node1.content[:40]}' vs '{node2.content[:40]}'",
                        confidence=0.6,
                        source_concepts=[node1.id, node2.id],
                        trigger="passive_integration",
                        implications=["Emotional or logical contradiction detected"]
                    )
                    self._register_insight(insight)

    def _content_overlap(self, content1: str, content2: str) -> float:
        """Calculate word overlap between two content strings"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / min(len(words1), len(words2))

    async def _attempt_insight_generation(self):
        """
        Actively try to generate insights from current state.
        """
        # Check for pattern insights from clusters
        clusters = self.semantic_network.cluster_by_activation(min_size=3)

        for cluster in clusters:
            if len(cluster) >= 4:
                nodes = [self.semantic_network.nodes[nid] for nid in cluster if nid in self.semantic_network.nodes]

                # Check if cluster has consistent emotional signature
                valences = [n.valence for n in nodes]
                if valences:
                    valence_std = self._std(valences)

                    if valence_std < 0.2:  # Consistent emotional tone
                        avg_valence = sum(valences) / len(valences)
                        tone = "positive" if avg_valence > 0.3 else "negative" if avg_valence < -0.3 else "neutral"

                        insight = Insight(
                            id=f"pattern_{len(self.insights)}",
                            insight_type=InsightType.PATTERN,
                            content=f"Cluster of {len(cluster)} related concepts with {tone} emotional tone",
                            confidence=0.5 + (1 - valence_std) * 0.3,
                            source_concepts=list(cluster),
                            trigger="pattern_detection",
                            implications=[f"These concepts form a coherent {tone} schema"]
                        )
                        self._register_insight(insight)

    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)

    def _register_insight(self, insight: Insight):
        """Register a new insight and notify listeners"""
        # Avoid duplicate insights
        for existing in self.insights[-20:]:  # Check recent
            if (existing.insight_type == insight.insight_type and
                self._content_overlap(existing.content, insight.content) > 0.7):
                return

        self.insights.append(insight)

        if self.on_insight:
            self.on_insight(insight)

    def update_predictive_model(self, domain: str, observation: str,
                                confirmed: bool, prediction_id: str = None):
        """
        Update predictive models based on observations.
        This is how the dreamspace learns about the world.
        """
        if domain not in self.predictive_models:
            self.predictive_models[domain] = PredictiveModel(
                id=f"model_{domain}",
                domain=domain,
                hypothesis=f"Model for {domain}",
                evidence_for=[],
                evidence_against=[],
                confidence=0.5,
                predictions=[]
            )

        model = self.predictive_models[domain]

        if confirmed:
            model.evidence_for.append(observation)
            model.confidence = min(0.95, model.confidence + 0.05)
            model.accuracy_history.append(1.0)
        else:
            model.evidence_against.append(observation)
            model.confidence = max(0.1, model.confidence - 0.1)
            model.accuracy_history.append(0.0)

        model.last_updated = datetime.now()

        # Keep history bounded
        if len(model.accuracy_history) > 100:
            model.accuracy_history = model.accuracy_history[-100:]

    async def generate_counterfactual(self, scenario: str,
                                      what_if: str) -> Dict[str, Any]:
        """
        Generate a counterfactual simulation.
        What would happen if X were different?
        """
        self.state = DreamState.SIMULATING

        # Find relevant concepts
        scenario_words = set(scenario.lower().split())
        what_if_words = set(what_if.lower().split())

        relevant = []
        for node_id, node in self.semantic_network.nodes.items():
            node_words = set(node.content.lower().split())
            if node_words & scenario_words or node_words & what_if_words:
                relevant.append(node)

        # Build counterfactual scenario
        original_states = {n.id: (n.valence, n.activation) for n in relevant}

        # Simulate the change
        changed_concepts = []
        for node in relevant:
            if set(node.content.lower().split()) & what_if_words:
                # This concept is directly affected
                node.valence = -node.valence  # Flip valence as simple simulation
                node.activation = 0.9
                changed_concepts.append(node.id)

        # Spread effects
        if changed_concepts:
            affected = self.semantic_network.spread_activation(
                changed_concepts,
                initial_activation=0.8,
                decay=0.3,
                max_steps=5
            )
        else:
            affected = []

        # Restore original states
        for node in relevant:
            if node.id in original_states:
                node.valence, node.activation = original_states[node.id]

        # Generate result
        fragment = DreamFragment(
            id=f"counterfactual_{self._cycle_count}",
            content=f"If {what_if}, then {len(affected)} concepts would be affected",
            source_concepts=changed_concepts,
            dream_state=DreamState.SIMULATING,
            coherence=0.5,
            novelty=0.8,
            emotional_charge=0.5
        )
        self.dream_fragments.append(fragment)

        return {
            "scenario": scenario,
            "what_if": what_if,
            "directly_changed": changed_concepts,
            "cascade_effects": [(nid, act) for nid, act in affected if nid not in changed_concepts],
            "fragment": fragment
        }

    async def creative_synthesis(self, seed_concepts: List[str],
                                temperature: float = None) -> Dict[str, Any]:
        """
        Attempt creative synthesis from seed concepts.
        Generates novel combinations.
        """
        self.state = DreamState.CREATING
        temp = temperature or self.creativity_temperature

        # Find seed nodes
        seed_nodes = []
        for concept in seed_concepts:
            for node_id, node in self.semantic_network.nodes.items():
                if concept.lower() in node.content.lower():
                    seed_nodes.append(node_id)
                    break

        if len(seed_nodes) < 2:
            return {"error": "Not enough seed concepts found in network"}

        # Spread activation from all seeds
        activated = self.semantic_network.spread_activation(
            seed_nodes,
            initial_activation=1.0,
            decay=0.1 * temp,  # Higher temp = less decay = more spread
            threshold=0.05,
            max_steps=10
        )

        # Find intersection points (concepts activated by multiple seeds)
        activation_sources: Dict[str, Set[str]] = defaultdict(set)

        for seed_id in seed_nodes:
            single_spread = self.semantic_network.spread_activation(
                [seed_id],
                initial_activation=1.0,
                decay=0.2,
                threshold=0.2,
                max_steps=5
            )
            for node_id, _ in single_spread:
                activation_sources[node_id].add(seed_id)

        # Find nodes activated by multiple sources
        intersection_nodes = [
            node_id for node_id, sources in activation_sources.items()
            if len(sources) >= 2
        ]

        # The intersection nodes are the creative synthesis points
        synthesis = []
        for node_id in intersection_nodes:
            if node_id in self.semantic_network.nodes:
                node = self.semantic_network.nodes[node_id]
                synthesis.append({
                    "concept": node.content,
                    "connecting_seeds": list(activation_sources[node_id]),
                    "novelty": 1.0 - (node.access_count / max(1, self._cycle_count))
                })

        # Sort by novelty
        synthesis.sort(key=lambda x: x["novelty"], reverse=True)

        if synthesis:
            insight = Insight(
                id=f"synthesis_{len(self.insights)}",
                insight_type=InsightType.SYNTHESIS,
                content=f"Creative combination: {synthesis[0]['concept'][:50]} bridges {seed_concepts}",
                confidence=0.5 + len(intersection_nodes) * 0.05,
                source_concepts=seed_nodes + intersection_nodes[:3],
                trigger="creative_synthesis",
                implications=[f"Found {len(intersection_nodes)} bridging concepts"]
            )
            self._register_insight(insight)

        fragment = DreamFragment(
            id=f"creative_{self._cycle_count}",
            content=f"Synthesizing {seed_concepts} -> {len(synthesis)} novel combinations",
            source_concepts=seed_nodes,
            dream_state=DreamState.CREATING,
            coherence=0.3 + (1 - temp) * 0.4,
            novelty=temp,
            emotional_charge=0.6
        )
        self.dream_fragments.append(fragment)

        return {
            "seeds": seed_concepts,
            "synthesis_points": synthesis[:10],
            "total_activated": len(activated),
            "fragment": fragment
        }

    def get_recent_insights(self, n: int = 10) -> List[Insight]:
        """Get the n most recent insights"""
        return self.insights[-n:]

    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get summary of emotional processing"""
        if not self.emotional_buffer:
            return {"status": "no_emotional_content"}

        recent = self.emotional_buffer[-20:]

        avg_valence = sum(e["valence"] for e in recent) / len(recent)
        avg_arousal = sum(e["arousal"] for e in recent) / len(recent)

        return {
            "average_valence": avg_valence,
            "average_arousal": avg_arousal,
            "emotional_state": self._classify_emotion(avg_valence, avg_arousal),
            "recent_experiences": len(recent),
            "high_arousal_events": sum(1 for e in recent if e["arousal"] > 0.7)
        }

    def _classify_emotion(self, valence: float, arousal: float) -> str:
        """Classify emotion based on valence-arousal model"""
        if valence > 0.3:
            if arousal > 0.5:
                return "excited/joyful"
            else:
                return "calm/content"
        elif valence < -0.3:
            if arousal > 0.5:
                return "angry/anxious"
            else:
                return "sad/depressed"
        else:
            if arousal > 0.5:
                return "alert/surprised"
            else:
                return "neutral/relaxed"

    def export_state(self) -> Dict[str, Any]:
        """Export the complete dreamspace state"""
        return {
            "state": self.state.value,
            "cycle_count": self._cycle_count,
            "network_size": len(self.semantic_network.nodes),
            "total_connections": sum(
                len(n.connections) for n in self.semantic_network.nodes.values()
            ),
            "insights": [i.to_dict() for i in self.insights],
            "dream_fragments": len(self.dream_fragments),
            "incubating_problems": len(self.incubating_problems),
            "predictive_models": {
                k: {
                    "domain": m.domain,
                    "confidence": m.confidence,
                    "evidence_for": len(m.evidence_for),
                    "evidence_against": len(m.evidence_against)
                }
                for k, m in self.predictive_models.items()
            },
            "emotional_summary": self.get_emotional_summary()
        }


# Convenience functions
def create_dreamspace(agi_system: Optional[Any] = None) -> Dreamspace:
    """Create a new dreamspace instance"""
    return Dreamspace(agi_system)


async def demo_dreamspace():
    """Demonstrate dreamspace capabilities"""
    ds = create_dreamspace()

    # Add some concepts
    concepts = [
        ("AI will transform healthcare", "belief", 0.6, 0.5),
        ("Privacy concerns are growing", "observation", -0.4, 0.6),
        ("Machine learning enables diagnosis", "fact", 0.3, 0.3),
        ("Data collection raises ethical issues", "concern", -0.5, 0.7),
        ("Automation improves efficiency", "belief", 0.5, 0.4),
        ("Jobs may be displaced", "fear", -0.7, 0.8),
        ("New opportunities will emerge", "hope", 0.7, 0.6),
        ("Regulation is needed", "opinion", 0.0, 0.5),
    ]

    concept_ids = []
    for content, ctype, valence, arousal in concepts:
        cid = ds.ingest_experience(content, ctype, valence, arousal)
        concept_ids.append(cid)

    # Run some cycles manually
    for _ in range(5):
        await ds._consolidate_memories()
        await ds._mind_wander()
        await ds._attempt_insight_generation()

    # Try creative synthesis
    result = await ds.creative_synthesis(["AI", "healthcare", "privacy"])

    # Get insights
    insights = ds.get_recent_insights()

    print("=== Dreamspace Demo ===")
    print(f"Network: {len(ds.semantic_network.nodes)} nodes")
    print(f"Insights: {len(insights)}")
    for insight in insights:
        print(f"  - [{insight.insight_type.value}] {insight.content}")
    print(f"\nCreative synthesis result: {len(result.get('synthesis_points', []))} combinations")
    print(f"Emotional state: {ds.get_emotional_summary()['emotional_state']}")

    return ds


if __name__ == "__main__":
    asyncio.run(demo_dreamspace())
