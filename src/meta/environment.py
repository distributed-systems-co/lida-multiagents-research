"""Constructive environment for evolving meta-capabilities.

The environment provides:
- Mutation operators for structures
- Evolution through selection and recombination
- Caching of novel capabilities in the hypergraph
- Interactive construction of new agent types
"""

from __future__ import annotations

import copy
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, Optional, Tuple

from .structures import (
    MetaStructure, StructureType, Capability, CapabilityType,
    compose_structures, apply_meta,
)
from .templates import AgentTemplate, TemplateParameter
from .hypergraph import (
    Hypergraph, HyperNode, HyperEdge, EdgeType, get_capability_graph,
)

logger = logging.getLogger(__name__)


class MutationType(str, Enum):
    """Types of mutations that can be applied."""
    ADD_CAPABILITY = "add_capability"
    REMOVE_CAPABILITY = "remove_capability"
    MODIFY_CAPABILITY = "modify_capability"
    ADD_STRUCTURE = "add_structure"
    REMOVE_STRUCTURE = "remove_structure"
    SWAP_STRUCTURE = "swap_structure"
    COMPOSE = "compose"
    DECOMPOSE = "decompose"
    PARAMETERIZE = "parameterize"
    SPECIALIZE = "specialize"
    GENERALIZE = "generalize"
    MERGE = "merge"
    SPLIT = "split"
    REWIRE = "rewire"


@dataclass
class Mutation:
    """A mutation operation on a structure."""

    mutation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    mutation_type: MutationType = MutationType.MODIFY_CAPABILITY
    target_id: Optional[str] = None  # Target structure/capability ID
    parameters: dict = field(default_factory=dict)

    # For tracking
    applied_at: Optional[datetime] = None
    success: bool = False
    result_id: Optional[str] = None

    def apply(self, structure: MetaStructure) -> MetaStructure:
        """Apply this mutation to a structure."""
        result = copy.deepcopy(structure)
        result.version += 1

        try:
            if self.mutation_type == MutationType.ADD_CAPABILITY:
                cap = self.parameters.get("capability")
                if cap:
                    result.capabilities.append(cap)

            elif self.mutation_type == MutationType.REMOVE_CAPABILITY:
                cap_id = self.parameters.get("capability_id")
                result.capabilities = [
                    c for c in result.capabilities
                    if c.capability_id != cap_id
                ]

            elif self.mutation_type == MutationType.MODIFY_CAPABILITY:
                cap_id = self.parameters.get("capability_id")
                modifications = self.parameters.get("modifications", {})
                for cap in result.capabilities:
                    if cap.capability_id == cap_id:
                        for key, value in modifications.items():
                            setattr(cap, key, value)

            elif self.mutation_type == MutationType.ADD_STRUCTURE:
                sub = self.parameters.get("structure")
                if sub:
                    result.sub_structures.append(sub)

            elif self.mutation_type == MutationType.REMOVE_STRUCTURE:
                struct_id = self.parameters.get("structure_id")
                result.sub_structures = [
                    s for s in result.sub_structures
                    if s.structure_id != struct_id
                ]

            elif self.mutation_type == MutationType.COMPOSE:
                other = self.parameters.get("other")
                if other:
                    result = result.compose(other)

            elif self.mutation_type == MutationType.PARAMETERIZE:
                # Convert to parametric structure
                params = self.parameters.get("parameters", {})
                result.structure_type = StructureType.PARAMETRIC
                result.parameters.update(params)

            elif self.mutation_type == MutationType.SPECIALIZE:
                # Apply specific parameter values
                params = self.parameters.get("specialization", {})
                result = result.instantiate(**params)

            elif self.mutation_type == MutationType.MERGE:
                # Merge capabilities from another structure
                other = self.parameters.get("other")
                if other:
                    for cap in other.capabilities:
                        if cap not in result.capabilities:
                            result.capabilities.append(copy.deepcopy(cap))

            elif self.mutation_type == MutationType.REWIRE:
                # Change capability connections
                rewiring = self.parameters.get("rewiring", {})
                for cap in result.capabilities:
                    if cap.capability_id in rewiring:
                        new_outputs = rewiring[cap.capability_id]
                        cap.outputs = new_outputs

            self.success = True
            self.result_id = result.structure_id
            self.applied_at = datetime.now()

        except Exception as e:
            logger.error(f"Mutation {self.mutation_id} failed: {e}")
            self.success = False

        result.emerged_from.append(structure.structure_id)
        result.metadata["mutation"] = {
            "type": self.mutation_type.value,
            "mutation_id": self.mutation_id,
        }

        return result


class ConstructiveEnvironment:
    """Environment for constructing and evolving meta-capabilities.

    Provides:
    - Mutation operators
    - Composition operators
    - Evolution through selection
    - Caching in hypergraph
    """

    def __init__(self, hypergraph: Optional[Hypergraph] = None):
        self.graph = hypergraph or get_capability_graph()
        self._mutation_history: list[Mutation] = []
        self._population: dict[str, MetaStructure] = {}
        self._fitness_scores: dict[str, float] = {}

        # Registered mutation operators
        self._mutation_operators: dict[MutationType, Callable] = {}

        # Evolution parameters
        self.population_size = 100
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        self.selection_pressure = 2.0

        # Register default operators
        self._register_default_operators()

    def _register_default_operators(self):
        """Register default mutation operators."""
        self._mutation_operators[MutationType.ADD_CAPABILITY] = self._mutate_add_cap
        self._mutation_operators[MutationType.REMOVE_CAPABILITY] = self._mutate_remove_cap
        self._mutation_operators[MutationType.COMPOSE] = self._mutate_compose
        self._mutation_operators[MutationType.MERGE] = self._mutate_merge

    def _mutate_add_cap(self, structure: MetaStructure) -> Mutation:
        """Create a mutation to add a random capability."""
        cap_types = list(CapabilityType)
        cap = Capability(
            name=f"generated_{random.randint(1000, 9999)}",
            capability_type=random.choice(cap_types),
            inputs=["input"],
            outputs=["output"],
        )
        return Mutation(
            mutation_type=MutationType.ADD_CAPABILITY,
            target_id=structure.structure_id,
            parameters={"capability": cap},
        )

    def _mutate_remove_cap(self, structure: MetaStructure) -> Optional[Mutation]:
        """Create a mutation to remove a capability."""
        if not structure.capabilities:
            return None
        cap = random.choice(structure.capabilities)
        return Mutation(
            mutation_type=MutationType.REMOVE_CAPABILITY,
            target_id=structure.structure_id,
            parameters={"capability_id": cap.capability_id},
        )

    def _mutate_compose(self, structure: MetaStructure) -> Optional[Mutation]:
        """Create a mutation to compose with another structure."""
        # Find a compatible structure from population
        candidates = []
        for other in self._population.values():
            if other.structure_id != structure.structure_id:
                if structure.can_compose_with(other):
                    candidates.append(other)

        if not candidates:
            return None

        other = random.choice(candidates)
        return Mutation(
            mutation_type=MutationType.COMPOSE,
            target_id=structure.structure_id,
            parameters={"other": other},
        )

    def _mutate_merge(self, structure: MetaStructure) -> Optional[Mutation]:
        """Create a mutation to merge with another structure."""
        if len(self._population) < 2:
            return None

        others = [s for s in self._population.values()
                  if s.structure_id != structure.structure_id]
        if not others:
            return None

        other = random.choice(others)
        return Mutation(
            mutation_type=MutationType.MERGE,
            target_id=structure.structure_id,
            parameters={"other": other},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Construction Operations
    # ─────────────────────────────────────────────────────────────────────────

    def create_structure(
        self,
        name: str,
        capabilities: list[Capability] = None,
        structure_type: StructureType = StructureType.COMPOSITE,
    ) -> MetaStructure:
        """Create a new structure and add to environment."""
        structure = MetaStructure(
            name=name,
            structure_type=structure_type,
            capabilities=capabilities or [],
        )

        self._population[structure.structure_id] = structure
        self.graph.add_structure(structure)

        return structure

    def compose(
        self,
        *structures: MetaStructure,
        cache: bool = True,
    ) -> MetaStructure:
        """Compose multiple structures."""
        # Check cache first
        source_ids = [s.structure_id for s in structures]
        if cache:
            cached = self.graph.get_cached_composition(source_ids)
            if cached:
                return cached

        result = compose_structures(*structures)
        self._population[result.structure_id] = result

        # Cache and add to graph
        self.graph.add_structure(result)
        if cache:
            self.graph.cache_capability_composition(source_ids, result)

        # Create composition edge
        self.graph.create_edge(
            sources=source_ids,
            targets=[result.structure_id],
            edge_type=EdgeType.COMPOSITION,
        )

        return result

    def mutate(
        self,
        structure: MetaStructure,
        mutation_type: Optional[MutationType] = None,
    ) -> MetaStructure:
        """Apply a mutation to a structure."""
        if mutation_type is None:
            # Random mutation
            mutation_type = random.choice(list(MutationType))

        # Get or create mutation
        if mutation_type in self._mutation_operators:
            mutation = self._mutation_operators[mutation_type](structure)
            if mutation is None:
                return structure
        else:
            mutation = Mutation(
                mutation_type=mutation_type,
                target_id=structure.structure_id,
            )

        # Apply mutation
        result = mutation.apply(structure)
        self._mutation_history.append(mutation)

        # Add to population and graph
        self._population[result.structure_id] = result
        self.graph.add_structure(result)

        # Create derivation edge
        self.graph.create_edge(
            sources=[structure.structure_id],
            targets=[result.structure_id],
            edge_type=EdgeType.DERIVATION,
            data={"mutation_type": mutation_type.value},
        )

        return result

    def apply_recursive(
        self,
        structure: MetaStructure,
        target: Any,
        max_depth: int = 10,
    ) -> Any:
        """Recursively apply a structure until fixed point."""
        result = target
        seen_results = set()

        for depth in range(max_depth):
            # Apply structure
            new_result = apply_meta(structure, result)

            # Check for fixed point
            result_sig = str(new_result) if not isinstance(new_result, MetaStructure) else new_result.signature()
            if result_sig in seen_results:
                break

            seen_results.add(result_sig)
            result = new_result

            # If result is a structure, potentially recurse
            if isinstance(result, MetaStructure) and result.structure_type == StructureType.RECURSIVE:
                result = result.recurse(depth)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Evolution Operations
    # ─────────────────────────────────────────────────────────────────────────

    def set_fitness(self, structure_id: str, fitness: float):
        """Set the fitness score for a structure."""
        self._fitness_scores[structure_id] = fitness

    def evaluate_fitness(
        self,
        structure: MetaStructure,
        evaluator: Callable[[MetaStructure], float],
    ) -> float:
        """Evaluate and store fitness for a structure."""
        fitness = evaluator(structure)
        self._fitness_scores[structure.structure_id] = fitness
        return fitness

    def select(
        self,
        n: int = 1,
        method: str = "tournament",
    ) -> list[MetaStructure]:
        """Select structures based on fitness."""
        if not self._population:
            return []

        structures = list(self._population.values())

        if method == "tournament":
            selected = []
            for _ in range(n):
                # Tournament selection
                tournament_size = min(int(self.selection_pressure), len(structures))
                tournament = random.sample(structures, tournament_size)
                winner = max(tournament,
                           key=lambda s: self._fitness_scores.get(s.structure_id, 0))
                selected.append(winner)
            return selected

        elif method == "roulette":
            # Roulette wheel selection
            total_fitness = sum(
                self._fitness_scores.get(s.structure_id, 0.01)
                for s in structures
            )
            selected = []
            for _ in range(n):
                pick = random.uniform(0, total_fitness)
                current = 0
                for s in structures:
                    current += self._fitness_scores.get(s.structure_id, 0.01)
                    if current >= pick:
                        selected.append(s)
                        break
            return selected

        else:
            # Random selection
            return random.sample(structures, min(n, len(structures)))

    def crossover(
        self,
        parent1: MetaStructure,
        parent2: MetaStructure,
    ) -> Tuple[MetaStructure, MetaStructure]:
        """Crossover two structures to produce offspring."""
        # Single-point crossover on capabilities
        p1_caps = list(parent1.capabilities)
        p2_caps = list(parent2.capabilities)

        if p1_caps and p2_caps:
            point1 = random.randint(0, len(p1_caps))
            point2 = random.randint(0, len(p2_caps))

            child1_caps = p1_caps[:point1] + p2_caps[point2:]
            child2_caps = p2_caps[:point2] + p1_caps[point1:]
        else:
            child1_caps = p1_caps + p2_caps
            child2_caps = p2_caps + p1_caps

        child1 = MetaStructure(
            name=f"{parent1.name}x{parent2.name}_1",
            structure_type=StructureType.COMPOSITE,
            capabilities=child1_caps,
            emerged_from=[parent1.structure_id, parent2.structure_id],
        )

        child2 = MetaStructure(
            name=f"{parent1.name}x{parent2.name}_2",
            structure_type=StructureType.COMPOSITE,
            capabilities=child2_caps,
            emerged_from=[parent1.structure_id, parent2.structure_id],
        )

        # Add to population and graph
        for child in [child1, child2]:
            self._population[child.structure_id] = child
            self.graph.add_structure(child)

        # Create emergence edges
        self.graph.create_edge(
            sources=[parent1.structure_id, parent2.structure_id],
            targets=[child1.structure_id, child2.structure_id],
            edge_type=EdgeType.EMERGENCE,
        )

        return child1, child2

    def evolve_generation(
        self,
        evaluator: Callable[[MetaStructure], float],
    ) -> dict:
        """Evolve one generation."""
        # Evaluate all structures
        for structure in self._population.values():
            if structure.structure_id not in self._fitness_scores:
                self.evaluate_fitness(structure, evaluator)

        stats = {
            "generation_size": len(self._population),
            "best_fitness": max(self._fitness_scores.values()) if self._fitness_scores else 0,
            "avg_fitness": sum(self._fitness_scores.values()) / len(self._fitness_scores) if self._fitness_scores else 0,
            "mutations": 0,
            "crossovers": 0,
        }

        new_population = {}

        # Elitism: keep best
        elite_count = max(1, len(self._population) // 10)
        elite = sorted(
            self._population.values(),
            key=lambda s: self._fitness_scores.get(s.structure_id, 0),
            reverse=True,
        )[:elite_count]

        for e in elite:
            new_population[e.structure_id] = e

        # Fill rest with offspring
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate and len(self._population) >= 2:
                parents = self.select(2, method="tournament")
                child1, child2 = self.crossover(parents[0], parents[1])
                new_population[child1.structure_id] = child1
                stats["crossovers"] += 1
            else:
                parent = self.select(1)[0]
                child = self.mutate(parent)
                new_population[child.structure_id] = child
                stats["mutations"] += 1

        self._population = new_population
        return stats

    # ─────────────────────────────────────────────────────────────────────────
    # Discovery and Emergence
    # ─────────────────────────────────────────────────────────────────────────

    def discover_emergent_capabilities(
        self,
        min_structures: int = 2,
    ) -> list[MetaStructure]:
        """Look for emergent capabilities from combinations."""
        discovered = []

        # Find structures that compose well
        for structure in self._population.values():
            candidates = self.graph.get_composition_candidates(structure.structure_id)

            for candidate_id, score in candidates[:5]:
                if score > 0.5:  # High compatibility
                    other_node = self.graph.get_node(candidate_id)
                    if other_node and isinstance(other_node.data, MetaStructure):
                        # Compose and check for emergence
                        composed = self.compose(structure, other_node.data)

                        # Check if composed has new capabilities
                        source_caps = set(
                            c.signature() for c in structure.capabilities
                        ) | set(
                            c.signature() for c in other_node.data.capabilities
                        )
                        composed_caps = set(
                            c.signature() for c in composed.capabilities
                        )

                        # Mark as emergent if something new appeared
                        if len(composed_caps) > len(source_caps):
                            composed.structure_type = StructureType.EMERGENT
                            composed.emergence_conditions = {
                                "sources": [structure.structure_id, candidate_id],
                                "compatibility_score": score,
                            }
                            discovered.append(composed)

        return discovered

    def generate_novel_capability(
        self,
        requirements: dict,
    ) -> Optional[MetaStructure]:
        """Generate a novel capability from requirements."""
        # Search for structures that match requirements
        candidates = []

        for structure in self._population.values():
            match_score = 0

            if "inputs" in requirements:
                for cap in structure.capabilities:
                    if set(requirements["inputs"]) & set(cap.inputs):
                        match_score += 1

            if "outputs" in requirements:
                for cap in structure.capabilities:
                    if set(requirements["outputs"]) & set(cap.outputs):
                        match_score += 1

            if "capability_types" in requirements:
                for cap in structure.capabilities:
                    if cap.capability_type in requirements["capability_types"]:
                        match_score += 1

            if match_score > 0:
                candidates.append((structure, match_score))

        if not candidates:
            return None

        # Compose best candidates
        candidates.sort(key=lambda x: -x[1])
        to_compose = [c[0] for c in candidates[:3]]

        if len(to_compose) < 2:
            return to_compose[0] if to_compose else None

        return self.compose(*to_compose)

    # ─────────────────────────────────────────────────────────────────────────
    # Introspection
    # ─────────────────────────────────────────────────────────────────────────

    def get_population(self) -> list[MetaStructure]:
        """Get current population."""
        return list(self._population.values())

    def get_mutation_history(self) -> list[Mutation]:
        """Get mutation history."""
        return self._mutation_history

    def get_stats(self) -> dict:
        """Get environment statistics."""
        return {
            "population_size": len(self._population),
            "mutations_applied": len(self._mutation_history),
            "graph_stats": self.graph.get_stats(),
            "fitness_evaluated": len(self._fitness_scores),
            "best_fitness": max(self._fitness_scores.values()) if self._fitness_scores else None,
        }


# Global environment instance
_environment: Optional[ConstructiveEnvironment] = None


def get_environment() -> ConstructiveEnvironment:
    """Get or create the global constructive environment."""
    global _environment
    if _environment is None:
        _environment = ConstructiveEnvironment()
    return _environment
