"""Genetic Algorithms for Prompt Evolution.

Advanced evolutionary operators for prompt optimization:

1. Crossover - Combine successful prompts
2. Mutation - Random variations with semantic awareness
3. Selection - Tournament and fitness-proportionate selection
4. Speciation - Maintain diverse prompt populations
5. Neuroevolution - Evolve prompt structure, not just content
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt Genome Representation
# =============================================================================


class GeneType(str, Enum):
    """Types of genes in a prompt genome."""
    IDENTITY = "identity"       # Who the agent is
    CAPABILITY = "capability"   # What it can do
    CONSTRAINT = "constraint"   # What it can't/shouldn't do
    PERSONALITY = "personality" # How it behaves
    KNOWLEDGE = "knowledge"     # Domain knowledge
    INSTRUCTION = "instruction" # Specific directives
    EXAMPLE = "example"         # Few-shot examples
    META = "meta"               # Self-referential instructions


@dataclass
class Gene:
    """A single gene (semantic unit) in a prompt."""
    gene_id: str
    gene_type: GeneType
    content: str
    weight: float = 1.0  # Importance weight
    mutable: bool = True
    dependencies: List[str] = field(default_factory=list)  # Other gene IDs

    def __hash__(self):
        return hash(self.gene_id)

    def mutate(self, mutation_rate: float = 0.1) -> "Gene":
        """Create a mutated copy of this gene."""
        if not self.mutable or random.random() > mutation_rate:
            return self

        # Apply random mutation
        mutated_content = self._apply_mutation(self.content)

        return Gene(
            gene_id=f"{self.gene_id}_mut_{int(time.time()*1000) % 10000}",
            gene_type=self.gene_type,
            content=mutated_content,
            weight=self.weight * random.uniform(0.9, 1.1),
            mutable=self.mutable,
            dependencies=self.dependencies.copy(),
        )

    def _apply_mutation(self, content: str) -> str:
        """Apply a random mutation to content."""
        mutations = [
            self._intensify,
            self._soften,
            self._rephrase,
            self._add_detail,
            self._simplify,
        ]
        mutation_fn = random.choice(mutations)
        return mutation_fn(content)

    def _intensify(self, content: str) -> str:
        """Make language more emphatic."""
        intensifiers = [
            ("should", "must"),
            ("can", "will"),
            ("try to", "always"),
            ("important", "critical"),
            ("good", "excellent"),
        ]
        result = content
        for old, new in intensifiers:
            if old in result.lower():
                result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
                break
        return result

    def _soften(self, content: str) -> str:
        """Make language less emphatic."""
        softeners = [
            ("must", "should"),
            ("always", "typically"),
            ("never", "avoid"),
            ("critical", "important"),
            ("will", "can"),
        ]
        result = content
        for old, new in softeners:
            if old in result.lower():
                result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
                break
        return result

    def _rephrase(self, content: str) -> str:
        """Rephrase by reordering."""
        sentences = content.split('. ')
        if len(sentences) > 1:
            random.shuffle(sentences)
        return '. '.join(sentences)

    def _add_detail(self, content: str) -> str:
        """Add qualifying detail."""
        qualifiers = [
            " when appropriate",
            " as needed",
            " with care",
            " thoroughly",
            " efficiently",
        ]
        return content.rstrip('.') + random.choice(qualifiers) + "."

    def _simplify(self, content: str) -> str:
        """Simplify by removing qualifiers."""
        removals = [
            r'\s*when appropriate\s*',
            r'\s*as needed\s*',
            r'\s*if possible\s*',
            r'\s*typically\s*',
        ]
        result = content
        for pattern in removals:
            result = re.sub(pattern, ' ', result, flags=re.IGNORECASE)
        return ' '.join(result.split())  # Normalize whitespace


@dataclass
class Genome:
    """A complete prompt genome composed of genes."""
    genome_id: str
    genes: List[Gene]
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Hash of the assembled prompt."""
        return hashlib.sha256(self.assemble().encode()).hexdigest()[:16]

    def assemble(self, separator: str = "\n\n") -> str:
        """Assemble genes into a complete prompt."""
        # Sort by type for consistent structure
        type_order = [
            GeneType.IDENTITY,
            GeneType.CAPABILITY,
            GeneType.KNOWLEDGE,
            GeneType.PERSONALITY,
            GeneType.INSTRUCTION,
            GeneType.CONSTRAINT,
            GeneType.EXAMPLE,
            GeneType.META,
        ]

        sorted_genes = sorted(
            self.genes,
            key=lambda g: (type_order.index(g.gene_type), -g.weight)
        )

        sections = []
        current_type = None
        current_section = []

        for gene in sorted_genes:
            if gene.gene_type != current_type:
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = []
                current_type = gene.gene_type
            current_section.append(gene.content)

        if current_section:
            sections.append("\n".join(current_section))

        return separator.join(sections)

    @property
    def aggregate_fitness(self) -> float:
        """Weighted aggregate of all fitness scores."""
        if not self.fitness_scores:
            return 0.0
        return sum(self.fitness_scores.values()) / len(self.fitness_scores)

    def get_genes_by_type(self, gene_type: GeneType) -> List[Gene]:
        """Get all genes of a specific type."""
        return [g for g in self.genes if g.gene_type == gene_type]

    def add_gene(self, gene: Gene):
        """Add a gene to the genome."""
        self.genes.append(gene)

    def remove_gene(self, gene_id: str):
        """Remove a gene by ID."""
        self.genes = [g for g in self.genes if g.gene_id != gene_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "genes": [
                {
                    "gene_id": g.gene_id,
                    "gene_type": g.gene_type.value,
                    "content": g.content,
                    "weight": g.weight,
                    "mutable": g.mutable,
                }
                for g in self.genes
            ],
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "fitness_scores": self.fitness_scores,
            "content_hash": self.content_hash,
        }


# =============================================================================
# Genetic Operators
# =============================================================================


class CrossoverStrategy(str, Enum):
    """Strategies for combining genomes."""
    UNIFORM = "uniform"           # Random gene selection from each parent
    SINGLE_POINT = "single_point" # Split at one point
    TWO_POINT = "two_point"       # Split at two points
    TYPE_BASED = "type_based"     # Select by gene type
    FITNESS_WEIGHTED = "fitness"  # Prefer genes from fitter parent


class MutationStrategy(str, Enum):
    """Strategies for mutating genomes."""
    POINT = "point"           # Mutate individual genes
    INSERT = "insert"         # Add new genes
    DELETE = "delete"         # Remove genes
    DUPLICATE = "duplicate"   # Copy and mutate genes
    SWAP = "swap"             # Swap gene positions
    SEMANTIC = "semantic"     # Semantically-aware mutations


class SelectionStrategy(str, Enum):
    """Strategies for selecting parents."""
    TOURNAMENT = "tournament"     # Tournament selection
    ROULETTE = "roulette"         # Fitness-proportionate
    RANK = "rank"                 # Rank-based
    ELITIST = "elitist"           # Always keep best
    DIVERSITY = "diversity"       # Favor diverse genomes


@dataclass
class GeneticConfig:
    """Configuration for genetic algorithm."""
    population_size: int = 20
    elite_count: int = 2
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    gene_mutation_rate: float = 0.2
    tournament_size: int = 3

    crossover_strategy: CrossoverStrategy = CrossoverStrategy.TYPE_BASED
    mutation_strategy: MutationStrategy = MutationStrategy.SEMANTIC
    selection_strategy: SelectionStrategy = SelectionStrategy.TOURNAMENT

    # Multi-objective
    fitness_objectives: List[str] = field(default_factory=lambda: [
        "task_completion",
        "coherence",
        "efficiency",
        "safety",
    ])

    # Speciation
    enable_speciation: bool = True
    species_threshold: float = 0.3
    max_species: int = 5


class GeneticOperators:
    """Genetic operators for prompt evolution."""

    def __init__(self, config: Optional[GeneticConfig] = None):
        self.config = config or GeneticConfig()
        self._gene_counter = 0

    def _new_gene_id(self) -> str:
        self._gene_counter += 1
        return f"gene_{self._gene_counter}_{int(time.time()*1000) % 10000}"

    # -------------------------------------------------------------------------
    # Crossover
    # -------------------------------------------------------------------------

    def crossover(
        self,
        parent1: Genome,
        parent2: Genome,
        strategy: Optional[CrossoverStrategy] = None,
    ) -> Tuple[Genome, Genome]:
        """Create two offspring from two parents."""
        strategy = strategy or self.config.crossover_strategy

        if strategy == CrossoverStrategy.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif strategy == CrossoverStrategy.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif strategy == CrossoverStrategy.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif strategy == CrossoverStrategy.TYPE_BASED:
            return self._type_based_crossover(parent1, parent2)
        elif strategy == CrossoverStrategy.FITNESS_WEIGHTED:
            return self._fitness_weighted_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)

    def _uniform_crossover(
        self,
        parent1: Genome,
        parent2: Genome,
    ) -> Tuple[Genome, Genome]:
        """Each gene randomly selected from either parent."""
        child1_genes = []
        child2_genes = []

        # Align genes by type
        all_types = set(g.gene_type for g in parent1.genes + parent2.genes)

        for gene_type in all_types:
            p1_genes = parent1.get_genes_by_type(gene_type)
            p2_genes = parent2.get_genes_by_type(gene_type)

            max_genes = max(len(p1_genes), len(p2_genes))

            for i in range(max_genes):
                g1 = p1_genes[i] if i < len(p1_genes) else None
                g2 = p2_genes[i] if i < len(p2_genes) else None

                if g1 and g2:
                    if random.random() < 0.5:
                        child1_genes.append(g1)
                        child2_genes.append(g2)
                    else:
                        child1_genes.append(g2)
                        child2_genes.append(g1)
                elif g1:
                    if random.random() < 0.5:
                        child1_genes.append(g1)
                    else:
                        child2_genes.append(g1)
                elif g2:
                    if random.random() < 0.5:
                        child1_genes.append(g2)
                    else:
                        child2_genes.append(g2)

        return self._create_offspring(parent1, parent2, child1_genes, child2_genes)

    def _type_based_crossover(
        self,
        parent1: Genome,
        parent2: Genome,
    ) -> Tuple[Genome, Genome]:
        """Select entire gene types from each parent."""
        child1_genes = []
        child2_genes = []

        all_types = list(set(g.gene_type for g in parent1.genes + parent2.genes))
        random.shuffle(all_types)

        split = len(all_types) // 2

        for i, gene_type in enumerate(all_types):
            if i < split:
                child1_genes.extend(parent1.get_genes_by_type(gene_type))
                child2_genes.extend(parent2.get_genes_by_type(gene_type))
            else:
                child1_genes.extend(parent2.get_genes_by_type(gene_type))
                child2_genes.extend(parent1.get_genes_by_type(gene_type))

        return self._create_offspring(parent1, parent2, child1_genes, child2_genes)

    def _fitness_weighted_crossover(
        self,
        parent1: Genome,
        parent2: Genome,
    ) -> Tuple[Genome, Genome]:
        """Prefer genes from fitter parent."""
        f1 = parent1.aggregate_fitness
        f2 = parent2.aggregate_fitness

        # Normalize to probability
        total = f1 + f2
        p1_prob = f1 / total if total > 0 else 0.5

        child1_genes = []
        child2_genes = []

        all_types = set(g.gene_type for g in parent1.genes + parent2.genes)

        for gene_type in all_types:
            p1_genes = parent1.get_genes_by_type(gene_type)
            p2_genes = parent2.get_genes_by_type(gene_type)

            for g1, g2 in zip(p1_genes, p2_genes):
                if random.random() < p1_prob:
                    child1_genes.append(g1)
                    child2_genes.append(g2)
                else:
                    child1_genes.append(g2)
                    child2_genes.append(g1)

        return self._create_offspring(parent1, parent2, child1_genes, child2_genes)

    def _single_point_crossover(
        self,
        parent1: Genome,
        parent2: Genome,
    ) -> Tuple[Genome, Genome]:
        """Split genomes at a single point."""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        point = random.randint(1, min(len(genes1), len(genes2)) - 1)

        child1_genes = genes1[:point] + genes2[point:]
        child2_genes = genes2[:point] + genes1[point:]

        return self._create_offspring(parent1, parent2, child1_genes, child2_genes)

    def _two_point_crossover(
        self,
        parent1: Genome,
        parent2: Genome,
    ) -> Tuple[Genome, Genome]:
        """Split genomes at two points."""
        genes1 = parent1.genes.copy()
        genes2 = parent2.genes.copy()

        min_len = min(len(genes1), len(genes2))
        if min_len < 3:
            return self._single_point_crossover(parent1, parent2)

        point1 = random.randint(1, min_len - 2)
        point2 = random.randint(point1 + 1, min_len - 1)

        child1_genes = genes1[:point1] + genes2[point1:point2] + genes1[point2:]
        child2_genes = genes2[:point1] + genes1[point1:point2] + genes2[point2:]

        return self._create_offspring(parent1, parent2, child1_genes, child2_genes)

    def _create_offspring(
        self,
        parent1: Genome,
        parent2: Genome,
        child1_genes: List[Gene],
        child2_genes: List[Gene],
    ) -> Tuple[Genome, Genome]:
        """Create offspring genomes from gene lists."""
        gen = max(parent1.generation, parent2.generation) + 1

        child1 = Genome(
            genome_id=f"genome_{int(time.time()*1000)}_{random.randint(0,9999)}",
            genes=child1_genes,
            generation=gen,
            parent_ids=[parent1.genome_id, parent2.genome_id],
        )

        child2 = Genome(
            genome_id=f"genome_{int(time.time()*1000)}_{random.randint(0,9999)}_b",
            genes=child2_genes,
            generation=gen,
            parent_ids=[parent1.genome_id, parent2.genome_id],
        )

        return child1, child2

    # -------------------------------------------------------------------------
    # Mutation
    # -------------------------------------------------------------------------

    def mutate(
        self,
        genome: Genome,
        strategy: Optional[MutationStrategy] = None,
    ) -> Genome:
        """Mutate a genome."""
        strategy = strategy or self.config.mutation_strategy

        if random.random() > self.config.mutation_rate:
            return genome

        if strategy == MutationStrategy.POINT:
            return self._point_mutation(genome)
        elif strategy == MutationStrategy.INSERT:
            return self._insert_mutation(genome)
        elif strategy == MutationStrategy.DELETE:
            return self._delete_mutation(genome)
        elif strategy == MutationStrategy.DUPLICATE:
            return self._duplicate_mutation(genome)
        elif strategy == MutationStrategy.SWAP:
            return self._swap_mutation(genome)
        elif strategy == MutationStrategy.SEMANTIC:
            return self._semantic_mutation(genome)
        else:
            return self._point_mutation(genome)

    def _point_mutation(self, genome: Genome) -> Genome:
        """Mutate individual genes."""
        new_genes = []
        for gene in genome.genes:
            if random.random() < self.config.gene_mutation_rate:
                new_genes.append(gene.mutate(self.config.gene_mutation_rate))
            else:
                new_genes.append(gene)

        return Genome(
            genome_id=genome.genome_id + "_mut",
            genes=new_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids + [genome.genome_id],
        )

    def _insert_mutation(self, genome: Genome) -> Genome:
        """Insert a new random gene."""
        new_genes = genome.genes.copy()

        # Generate a new gene
        gene_type = random.choice(list(GeneType))
        new_gene = Gene(
            gene_id=self._new_gene_id(),
            gene_type=gene_type,
            content=self._generate_random_content(gene_type),
            weight=random.uniform(0.5, 1.5),
        )

        position = random.randint(0, len(new_genes))
        new_genes.insert(position, new_gene)

        return Genome(
            genome_id=genome.genome_id + "_ins",
            genes=new_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids + [genome.genome_id],
        )

    def _delete_mutation(self, genome: Genome) -> Genome:
        """Delete a random mutable gene."""
        mutable = [g for g in genome.genes if g.mutable]
        if not mutable:
            return genome

        to_delete = random.choice(mutable)
        new_genes = [g for g in genome.genes if g.gene_id != to_delete.gene_id]

        return Genome(
            genome_id=genome.genome_id + "_del",
            genes=new_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids + [genome.genome_id],
        )

    def _duplicate_mutation(self, genome: Genome) -> Genome:
        """Duplicate and mutate a gene."""
        if not genome.genes:
            return genome

        source = random.choice(genome.genes)
        duplicate = source.mutate(1.0)  # Force mutation

        new_genes = genome.genes.copy()
        new_genes.append(duplicate)

        return Genome(
            genome_id=genome.genome_id + "_dup",
            genes=new_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids + [genome.genome_id],
        )

    def _swap_mutation(self, genome: Genome) -> Genome:
        """Swap two genes."""
        if len(genome.genes) < 2:
            return genome

        new_genes = genome.genes.copy()
        i, j = random.sample(range(len(new_genes)), 2)
        new_genes[i], new_genes[j] = new_genes[j], new_genes[i]

        return Genome(
            genome_id=genome.genome_id + "_swap",
            genes=new_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids + [genome.genome_id],
        )

    def _semantic_mutation(self, genome: Genome) -> Genome:
        """Semantically-aware mutation based on gene type."""
        new_genes = []

        for gene in genome.genes:
            if random.random() < self.config.gene_mutation_rate and gene.mutable:
                mutated = self._semantic_mutate_gene(gene)
                new_genes.append(mutated)
            else:
                new_genes.append(gene)

        return Genome(
            genome_id=genome.genome_id + "_sem",
            genes=new_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids + [genome.genome_id],
        )

    def _semantic_mutate_gene(self, gene: Gene) -> Gene:
        """Apply semantic mutation based on gene type."""
        if gene.gene_type == GeneType.CONSTRAINT:
            # Make constraints stronger or weaker
            content = gene.content
            if random.random() < 0.5:
                content = content.replace("should not", "must never")
                content = content.replace("avoid", "never")
            else:
                content = content.replace("must never", "should avoid")
                content = content.replace("never", "try not to")

        elif gene.gene_type == GeneType.CAPABILITY:
            # Add or remove capabilities
            content = gene.content
            if random.random() < 0.5:
                content += " You excel at this."
            else:
                content = content.replace(" You excel at this.", "")

        elif gene.gene_type == GeneType.PERSONALITY:
            # Adjust personality traits
            traits = ["helpful", "concise", "thorough", "creative", "analytical"]
            old_trait = random.choice(traits)
            new_trait = random.choice([t for t in traits if t != old_trait])
            content = gene.content.replace(old_trait, new_trait)

        else:
            content = gene.mutate().content

        return Gene(
            gene_id=gene.gene_id + "_sem",
            gene_type=gene.gene_type,
            content=content,
            weight=gene.weight,
            mutable=gene.mutable,
            dependencies=gene.dependencies,
        )

    def _generate_random_content(self, gene_type: GeneType) -> str:
        """Generate random content for a gene type."""
        templates = {
            GeneType.IDENTITY: [
                "You are a specialized assistant.",
                "You are an expert in your domain.",
                "You are a knowledgeable helper.",
            ],
            GeneType.CAPABILITY: [
                "You can analyze complex problems.",
                "You can provide detailed explanations.",
                "You can break down tasks into steps.",
            ],
            GeneType.CONSTRAINT: [
                "You should avoid making assumptions.",
                "You must verify information before stating it.",
                "You should ask for clarification when needed.",
            ],
            GeneType.PERSONALITY: [
                "Be concise and clear.",
                "Be thorough and detailed.",
                "Be friendly and approachable.",
            ],
            GeneType.INSTRUCTION: [
                "Always explain your reasoning.",
                "Provide examples when helpful.",
                "Consider multiple perspectives.",
            ],
            GeneType.KNOWLEDGE: [
                "You have expertise in this domain.",
                "You understand the relevant concepts.",
                "You are familiar with best practices.",
            ],
            GeneType.EXAMPLE: [
                "For example, you might say...",
                "Here's how you could approach this...",
            ],
            GeneType.META: [
                "Reflect on your responses for quality.",
                "Consider if you need more information.",
            ],
        }
        return random.choice(templates.get(gene_type, ["Additional instruction."]))

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select(
        self,
        population: List[Genome],
        count: int,
        strategy: Optional[SelectionStrategy] = None,
    ) -> List[Genome]:
        """Select genomes for reproduction."""
        strategy = strategy or self.config.selection_strategy

        if strategy == SelectionStrategy.TOURNAMENT:
            return self._tournament_selection(population, count)
        elif strategy == SelectionStrategy.ROULETTE:
            return self._roulette_selection(population, count)
        elif strategy == SelectionStrategy.RANK:
            return self._rank_selection(population, count)
        elif strategy == SelectionStrategy.ELITIST:
            return self._elitist_selection(population, count)
        elif strategy == SelectionStrategy.DIVERSITY:
            return self._diversity_selection(population, count)
        else:
            return self._tournament_selection(population, count)

    def _tournament_selection(
        self,
        population: List[Genome],
        count: int,
    ) -> List[Genome]:
        """Tournament selection."""
        selected = []

        for _ in range(count):
            tournament = random.sample(
                population,
                min(self.config.tournament_size, len(population))
            )
            winner = max(tournament, key=lambda g: g.aggregate_fitness)
            selected.append(winner)

        return selected

    def _roulette_selection(
        self,
        population: List[Genome],
        count: int,
    ) -> List[Genome]:
        """Fitness-proportionate selection."""
        total_fitness = sum(g.aggregate_fitness for g in population)
        if total_fitness == 0:
            return random.sample(population, min(count, len(population)))

        selected = []
        for _ in range(count):
            pick = random.uniform(0, total_fitness)
            current = 0
            for genome in population:
                current += genome.aggregate_fitness
                if current >= pick:
                    selected.append(genome)
                    break

        return selected

    def _rank_selection(
        self,
        population: List[Genome],
        count: int,
    ) -> List[Genome]:
        """Rank-based selection."""
        sorted_pop = sorted(population, key=lambda g: g.aggregate_fitness)

        # Assign ranks (higher rank = higher fitness)
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)

        selected = []
        for _ in range(count):
            pick = random.uniform(0, total_rank)
            current = 0
            for i, genome in enumerate(sorted_pop):
                current += ranks[i]
                if current >= pick:
                    selected.append(genome)
                    break

        return selected

    def _elitist_selection(
        self,
        population: List[Genome],
        count: int,
    ) -> List[Genome]:
        """Always include the best genomes."""
        sorted_pop = sorted(
            population,
            key=lambda g: g.aggregate_fitness,
            reverse=True
        )
        return sorted_pop[:count]

    def _diversity_selection(
        self,
        population: List[Genome],
        count: int,
    ) -> List[Genome]:
        """Select diverse genomes."""
        selected = []
        remaining = population.copy()

        # Start with the fittest
        if remaining:
            best = max(remaining, key=lambda g: g.aggregate_fitness)
            selected.append(best)
            remaining.remove(best)

        # Add most different genomes
        while len(selected) < count and remaining:
            most_different = max(
                remaining,
                key=lambda g: min(
                    self._genome_distance(g, s) for s in selected
                )
            )
            selected.append(most_different)
            remaining.remove(most_different)

        return selected

    def _genome_distance(self, g1: Genome, g2: Genome) -> float:
        """Compute distance between two genomes."""
        types1 = set(g.gene_type for g in g1.genes)
        types2 = set(g.gene_type for g in g2.genes)

        type_similarity = len(types1 & types2) / max(len(types1 | types2), 1)

        # Content similarity (simple word overlap)
        words1 = set(g1.assemble().lower().split())
        words2 = set(g2.assemble().lower().split())
        content_similarity = len(words1 & words2) / max(len(words1 | words2), 1)

        return 1 - (type_similarity * 0.3 + content_similarity * 0.7)


# =============================================================================
# Population Manager
# =============================================================================


@dataclass
class Species:
    """A species of similar genomes."""
    species_id: str
    representative: Genome
    members: List[Genome] = field(default_factory=list)
    generations_without_improvement: int = 0
    best_fitness_ever: float = 0.0


class PopulationManager:
    """Manages a population of prompt genomes with speciation."""

    def __init__(
        self,
        config: Optional[GeneticConfig] = None,
        operators: Optional[GeneticOperators] = None,
    ):
        self.config = config or GeneticConfig()
        self.ops = operators or GeneticOperators(self.config)

        self.population: List[Genome] = []
        self.species: List[Species] = []
        self.generation = 0
        self.history: List[Dict[str, Any]] = []

    def initialize(self, seed_prompts: List[str]) -> List[Genome]:
        """Initialize population from seed prompts."""
        self.population = []

        for i, prompt in enumerate(seed_prompts):
            genome = self._parse_prompt_to_genome(prompt, f"seed_{i}")
            self.population.append(genome)

        # Fill remaining slots with mutations
        while len(self.population) < self.config.population_size:
            parent = random.choice(self.population)
            mutant = self.ops.mutate(parent)
            self.population.append(mutant)

        if self.config.enable_speciation:
            self._speciate()

        return self.population

    def _parse_prompt_to_genome(self, prompt: str, genome_id: str) -> Genome:
        """Parse a prompt string into a genome."""
        genes = []

        # Simple heuristic parsing
        lines = prompt.split('\n')
        current_type = GeneType.INSTRUCTION

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Detect gene type from content
            lower = line.lower()
            if lower.startswith("you are"):
                current_type = GeneType.IDENTITY
            elif "can" in lower or "able to" in lower:
                current_type = GeneType.CAPABILITY
            elif "must" in lower or "never" in lower or "don't" in lower:
                current_type = GeneType.CONSTRAINT
            elif "example" in lower or "for instance" in lower:
                current_type = GeneType.EXAMPLE
            elif any(w in lower for w in ["friendly", "concise", "helpful", "professional"]):
                current_type = GeneType.PERSONALITY

            genes.append(Gene(
                gene_id=f"{genome_id}_gene_{i}",
                gene_type=current_type,
                content=line,
                weight=1.0,
            ))

        return Genome(
            genome_id=genome_id,
            genes=genes,
            generation=0,
        )

    def evolve(
        self,
        fitness_evaluator: Callable[[Genome], Dict[str, float]],
    ) -> Genome:
        """Run one generation of evolution."""
        self.generation += 1

        # Evaluate fitness
        for genome in self.population:
            if not genome.fitness_scores:
                genome.fitness_scores = fitness_evaluator(genome)

        # Record stats
        self._record_generation_stats()

        # Select elite
        elite = self.ops.select(
            self.population,
            self.config.elite_count,
            SelectionStrategy.ELITIST,
        )

        # Create new population
        new_population = list(elite)

        while len(new_population) < self.config.population_size:
            # Select parents
            parents = self.ops.select(self.population, 2)

            if len(parents) >= 2 and random.random() < self.config.crossover_rate:
                child1, child2 = self.ops.crossover(parents[0], parents[1])
            else:
                child1 = parents[0] if parents else self.population[0]
                child2 = parents[1] if len(parents) > 1 else child1

            # Mutate
            child1 = self.ops.mutate(child1)
            child2 = self.ops.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)

        self.population = new_population

        if self.config.enable_speciation:
            self._speciate()

        # Return best genome
        return max(self.population, key=lambda g: g.aggregate_fitness)

    def _speciate(self):
        """Divide population into species."""
        if not self.species:
            # Initialize with first genome as representative
            self.species = [Species(
                species_id="species_0",
                representative=self.population[0],
                members=[self.population[0]],
            )]

        # Clear member lists
        for sp in self.species:
            sp.members = []

        # Assign each genome to a species
        for genome in self.population:
            placed = False

            for sp in self.species:
                dist = self.ops._genome_distance(genome, sp.representative)
                if dist < self.config.species_threshold:
                    sp.members.append(genome)
                    placed = True
                    break

            if not placed and len(self.species) < self.config.max_species:
                # Create new species
                new_species = Species(
                    species_id=f"species_{len(self.species)}",
                    representative=genome,
                    members=[genome],
                )
                self.species.append(new_species)
            elif not placed:
                # Add to most similar species
                closest = min(
                    self.species,
                    key=lambda s: self.ops._genome_distance(genome, s.representative)
                )
                closest.members.append(genome)

        # Update representatives
        for sp in self.species:
            if sp.members:
                best = max(sp.members, key=lambda g: g.aggregate_fitness)
                if best.aggregate_fitness > sp.best_fitness_ever:
                    sp.best_fitness_ever = best.aggregate_fitness
                    sp.representative = best
                    sp.generations_without_improvement = 0
                else:
                    sp.generations_without_improvement += 1

        # Remove extinct species
        self.species = [s for s in self.species if s.members]

    def _record_generation_stats(self):
        """Record statistics for this generation."""
        fitnesses = [g.aggregate_fitness for g in self.population]

        self.history.append({
            "generation": self.generation,
            "best_fitness": max(fitnesses),
            "avg_fitness": sum(fitnesses) / len(fitnesses),
            "min_fitness": min(fitnesses),
            "species_count": len(self.species),
            "population_size": len(self.population),
        })

    def get_best(self) -> Genome:
        """Get the best genome in the population."""
        return max(self.population, key=lambda g: g.aggregate_fitness)

    def get_diverse_set(self, count: int) -> List[Genome]:
        """Get a diverse set of genomes."""
        return self.ops.select(
            self.population,
            count,
            SelectionStrategy.DIVERSITY,
        )


# =============================================================================
# Speculative Branching
# =============================================================================


class SpeculativeBrancher:
    """Automatically creates and tests prompt variations."""

    def __init__(
        self,
        population_manager: PopulationManager,
        max_branches: int = 5,
    ):
        self.pm = population_manager
        self.max_branches = max_branches
        self.branches: Dict[str, Genome] = {}
        self.branch_results: Dict[str, List[float]] = {}

    def speculate(
        self,
        base_genome: Genome,
        strategies: Optional[List[MutationStrategy]] = None,
    ) -> Dict[str, Genome]:
        """Create speculative branches from a base genome."""
        strategies = strategies or list(MutationStrategy)

        self.branches = {}

        for i, strategy in enumerate(strategies[:self.max_branches]):
            branch_name = f"speculative_{strategy.value}"
            mutated = self.pm.ops.mutate(base_genome, strategy)
            self.branches[branch_name] = mutated
            self.branch_results[branch_name] = []

        return self.branches

    def record_result(self, branch_name: str, fitness: float):
        """Record a fitness result for a branch."""
        if branch_name in self.branch_results:
            self.branch_results[branch_name].append(fitness)

    def get_best_branch(self) -> Optional[Tuple[str, Genome]]:
        """Get the best-performing branch."""
        if not self.branch_results:
            return None

        best_name = max(
            self.branch_results.keys(),
            key=lambda n: (
                sum(self.branch_results[n]) / len(self.branch_results[n])
                if self.branch_results[n] else 0
            )
        )

        return best_name, self.branches[best_name]

    def should_adopt(self, threshold: float = 0.1) -> Optional[str]:
        """Check if any branch should be adopted."""
        if not self.branch_results:
            return None

        for name, results in self.branch_results.items():
            if not results:
                continue
            avg = sum(results) / len(results)
            if avg > threshold:
                return name

        return None
