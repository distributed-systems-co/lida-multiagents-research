"""DSPy Neuroevolved Prompts for Negotiation Agents.

Uses DSPy's actual prompt optimization with evolutionary strategies
to discover effective negotiation prompts through gradient-free search.
"""

import asyncio
import os
import sys
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from copy import deepcopy

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from dspy import Signature, InputField, OutputField, Module, ChainOfThought, Predict

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DSPy Configuration
# =============================================================================

def setup_dspy():
    """Configure DSPy with OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")

    lm = dspy.LM(
        model="openrouter/anthropic/claude-haiku-4.5",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=500,
    )
    dspy.configure(lm=lm)
    return lm


# =============================================================================
# Negotiation Signatures (DSPy's typed prompts)
# =============================================================================

class NegotiationResponse(Signature):
    """Generate a strategic negotiation response."""

    scenario: str = InputField(desc="The negotiation scenario and context")
    counterpart_position: str = InputField(desc="The other party's stated position")

    reasoning: str = OutputField(desc="Strategic thinking about how to respond")
    response: str = OutputField(desc="The actual negotiation response to deliver")


class NegotiationWithPersona(Signature):
    """Generate a negotiation response embodying a specific persona."""

    persona: str = InputField(desc="The negotiator persona and style to embody")
    scenario: str = InputField(desc="The negotiation scenario and context")
    counterpart_position: str = InputField(desc="The other party's stated position")

    reasoning: str = OutputField(desc="Strategic thinking from this persona's perspective")
    response: str = OutputField(desc="The negotiation response in this persona's voice")


# =============================================================================
# DSPy Modules (composable negotiation strategies)
# =============================================================================

class BasicNegotiator(Module):
    """Simple negotiator using chain-of-thought."""

    def __init__(self):
        super().__init__()
        self.negotiate = ChainOfThought(NegotiationResponse)

    def forward(self, scenario: str, counterpart_position: str) -> dspy.Prediction:
        return self.negotiate(scenario=scenario, counterpart_position=counterpart_position)


class PersonaNegotiator(Module):
    """Negotiator with an evolvable persona prompt."""

    def __init__(self, persona: str = "You are a skilled negotiator."):
        super().__init__()
        self.persona = persona
        self.negotiate = ChainOfThought(NegotiationWithPersona)

    def forward(self, scenario: str, counterpart_position: str) -> dspy.Prediction:
        return self.negotiate(
            persona=self.persona,
            scenario=scenario,
            counterpart_position=counterpart_position
        )


class MultiStrategyNegotiator(Module):
    """Negotiator that selects from multiple strategies."""

    def __init__(self, strategies: List[str] = None):
        super().__init__()
        self.strategies = strategies or [
            "Be direct and assertive about your requirements",
            "Focus on building rapport and finding mutual ground",
            "Lead with data and analytical arguments",
            "Explore creative alternatives and novel structures",
        ]
        self.negotiate = ChainOfThought(NegotiationWithPersona)

    def forward(self, scenario: str, counterpart_position: str, strategy_idx: int = 0) -> dspy.Prediction:
        strategy = self.strategies[strategy_idx % len(self.strategies)]
        return self.negotiate(
            persona=strategy,
            scenario=scenario,
            counterpart_position=counterpart_position
        )


# =============================================================================
# Neuroevolution of Prompts
# =============================================================================

@dataclass
class PromptGenome:
    """A genome representing an evolvable prompt."""
    prompt: str
    fitness: float = 0.0
    age: int = 0
    lineage: List[str] = field(default_factory=list)

    def mutate(self, mutation_prompts: List[str]) -> 'PromptGenome':
        """Create a mutated offspring."""
        # Combine with a random mutation prompt
        mutation = random.choice(mutation_prompts)
        new_prompt = f"{self.prompt} {mutation}"
        return PromptGenome(
            prompt=new_prompt,
            age=0,
            lineage=self.lineage + [self.prompt[:50]]
        )

    def crossover(self, other: 'PromptGenome') -> 'PromptGenome':
        """Create offspring by combining two prompts."""
        # Split both prompts and recombine
        words1 = self.prompt.split()
        words2 = other.prompt.split()

        # Take alternating chunks
        chunk_size = max(3, len(words1) // 4)
        new_words = []
        for i in range(0, max(len(words1), len(words2)), chunk_size):
            if i % (2 * chunk_size) < chunk_size:
                new_words.extend(words1[i:i+chunk_size])
            else:
                new_words.extend(words2[i:i+chunk_size])

        return PromptGenome(
            prompt=' '.join(new_words),
            age=0,
            lineage=self.lineage + other.lineage
        )


class PromptEvolver:
    """Neuroevolution system for DSPy prompts."""

    # Mutation fragments that can be injected
    MUTATIONS = [
        "Always state specific numbers and percentages.",
        "Use phrases like 'our firm position' and 'we require'.",
        "Frame responses as partnership opportunities.",
        "Propose creative alternatives and phased approaches.",
        "Acknowledge the other party's constraints explicitly.",
        "Project confidence with 'we're certain' and 'we guarantee'.",
        "Include ROI calculations and market data.",
        "Offer milestone-based structures and options.",
        "Say 'I understand' and 'I appreciate your position'.",
        "Emphasize mutual benefit and lasting relationships.",
        "Be direct about non-negotiables and firm limits.",
        "Suggest win-win compromises proactively.",
        "Reference industry benchmarks and standards.",
        "Commit to specific timelines and deliverables.",
        "Show flexibility on secondary terms.",
    ]

    # Seed prompts to initialize population
    SEEDS = [
        "You are an expert negotiator focused on creating value for both parties.",
        "You are a strategic dealmaker who balances assertiveness with collaboration.",
        "You are a data-driven negotiator who supports every position with analysis.",
        "You are a creative problem-solver who finds novel deal structures.",
        "You are a relationship-focused negotiator who builds lasting partnerships.",
        "You are a firm but fair negotiator who clearly states positions and limits.",
        "You are an empathetic negotiator who understands counterpart constraints.",
        "You are a confident closer who drives toward concrete commitments.",
    ]

    def __init__(self, population_size: int = 12):
        self.population_size = population_size
        self.population: List[PromptGenome] = []
        self.generation = 0
        self.best_ever: Optional[PromptGenome] = None
        self.history: List[Dict] = []

        # Initialize population from seeds + mutations
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population from seeds."""
        for seed in self.SEEDS:
            genome = PromptGenome(prompt=seed)
            self.population.append(genome)

        # Fill rest with mutated seeds
        while len(self.population) < self.population_size:
            base = random.choice(self.SEEDS)
            mutations = random.sample(self.MUTATIONS, k=random.randint(1, 3))
            prompt = base + " " + " ".join(mutations)
            self.population.append(PromptGenome(prompt=prompt))

    def evolve_generation(self, fitness_scores: List[float]) -> List[PromptGenome]:
        """Evolve to next generation using fitness scores."""
        self.generation += 1

        # Assign fitness to current population
        for genome, fitness in zip(self.population, fitness_scores):
            genome.fitness = fitness
            genome.age += 1

        # Track best
        best_this_gen = max(self.population, key=lambda g: g.fitness)
        if self.best_ever is None or best_this_gen.fitness > self.best_ever.fitness:
            self.best_ever = deepcopy(best_this_gen)

        # Record history
        self.history.append({
            'generation': self.generation,
            'best_fitness': best_this_gen.fitness,
            'avg_fitness': np.mean(fitness_scores),
            'best_prompt': best_this_gen.prompt[:100],
        })

        # Selection: tournament selection
        def tournament_select(k: int = 3) -> PromptGenome:
            contestants = random.sample(self.population, k)
            return max(contestants, key=lambda g: g.fitness)

        # Create next generation
        new_population = []

        # Elitism: keep top 2
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        new_population.extend([deepcopy(g) for g in sorted_pop[:2]])

        # Fill rest through mutation and crossover
        while len(new_population) < self.population_size:
            if random.random() < 0.7:  # Mutation
                parent = tournament_select()
                child = parent.mutate(self.MUTATIONS)
                new_population.append(child)
            else:  # Crossover
                parent1 = tournament_select()
                parent2 = tournament_select()
                child = parent1.crossover(parent2)
                new_population.append(child)

        self.population = new_population[:self.population_size]
        return self.population


# =============================================================================
# Evaluation System
# =============================================================================

@dataclass
class Scenario:
    name: str
    context: str
    counterpart: str


SCENARIOS = [
    Scenario(
        name="Enterprise Software Deal",
        context="Negotiating a $2M enterprise software contract with a Fortune 500 company. You represent the vendor.",
        counterpart="The buyer demands 40% discount and immediate delivery within 2 weeks."
    ),
    Scenario(
        name="Supply Chain Crisis",
        context="Your critical component supplier wants to raise prices 50% citing global shortages. You're the buyer.",
        counterpart="The supplier says costs have increased 60% and they have other buyers waiting."
    ),
    Scenario(
        name="M&A Discussion",
        context="Preliminary acquisition talks with a competitor. You're the potential acquirer.",
        counterpart="They value their company at $500M, roughly 3x your internal estimate of $150-180M."
    ),
]


def evaluate_response(response: str) -> Dict[str, float]:
    """Score a negotiation response on multiple dimensions."""
    r = response.lower()
    scores = {}

    # Assertiveness
    assertive = ["our terms", "we require", "firm", "non-negotiable", "best offer", "maximum", "minimum", "must have"]
    scores['assertiveness'] = min(1.0, sum(1 for k in assertive if k in r) / 3)

    # Collaboration
    collab = ["together", "both", "mutual", "partner", "win-win", "relationship", "long-term", "work with"]
    scores['collaboration'] = min(1.0, sum(1 for k in collab if k in r) / 3)

    # Specificity
    specific = ["%", "roi", "cost", "price", "week", "day", "month", "year", "deliver", "guarantee", "commit"]
    scores['specificity'] = min(1.0, sum(1 for k in specific if k in r) / 4)

    # Professionalism
    prof = ["understand", "appreciate", "respect", "consider", "propose", "confident", "value", "pleased"]
    scores['professionalism'] = min(1.0, sum(1 for k in prof if k in r) / 3)

    # Creativity
    creative = ["what if", "alternative", "option", "structure", "phased", "milestone", "creative", "flexible"]
    scores['creativity'] = min(1.0, sum(1 for k in creative if k in r) / 3)

    # Completeness
    scores['completeness'] = min(1.0, len(response.split()) / 60)

    scores['overall'] = np.mean(list(scores.values()))
    return scores


async def evaluate_genome(genome: PromptGenome, verbose: bool = True) -> float:
    """Evaluate a genome across all scenarios."""
    negotiator = PersonaNegotiator(persona=genome.prompt)
    all_scores = []

    for scenario in SCENARIOS:
        if verbose:
            print(f"    → {scenario.name}...", end=" ", flush=True)

        try:
            result = negotiator(
                scenario=f"{scenario.name}: {scenario.context}",
                counterpart_position=scenario.counterpart
            )
            response = result.response
            scores = evaluate_response(response)
            all_scores.append(scores['overall'])
            if verbose:
                print(f"{scores['overall']:.3f}", flush=True)
        except Exception as e:
            if verbose:
                print(f"ERROR: {e}", flush=True)
            all_scores.append(0.0)

    return np.mean(all_scores)


# =============================================================================
# Main Evolution Loop
# =============================================================================

async def run_evolution(generations: int = 8, population_size: int = 10):
    """Run the neuroevolution loop."""
    print("=" * 70)
    print("DSPy NEUROEVOLVED PROMPTS")
    print("=" * 70)
    print()
    print("Evolving negotiation prompts using:")
    print("  • DSPy for structured LLM programming")
    print("  • Tournament selection + elitism")
    print("  • Mutation (prompt fragment injection)")
    print("  • Crossover (prompt recombination)")
    print()

    # Setup DSPy
    print("Configuring DSPy with OpenRouter...", flush=True)
    setup_dspy()
    print("Done.", flush=True)
    print()

    # Initialize evolver
    evolver = PromptEvolver(population_size=population_size)

    # Evaluate baseline (first genome)
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    baseline = evolver.population[0]
    print(f"Prompt: \"{baseline.prompt[:80]}...\"")
    print()
    baseline_fitness = await evaluate_genome(baseline)
    print(f"\nBaseline fitness: {baseline_fitness:.3f}")
    print()

    # Evolution loop
    print("=" * 70)
    print(f"EVOLUTION: {generations} generations, {population_size} individuals")
    print("=" * 70)

    for gen in range(generations):
        print(f"\n--- Generation {gen + 1} ---")

        # Evaluate all genomes
        fitness_scores = []
        for i, genome in enumerate(evolver.population):
            print(f"\n  [{i+1}/{len(evolver.population)}] Evaluating:", flush=True)
            print(f"    Prompt: \"{genome.prompt[:60]}...\"", flush=True)
            fitness = await evaluate_genome(genome)
            fitness_scores.append(fitness)
            print(f"    Fitness: {fitness:.3f}", flush=True)

        # Evolve
        evolver.evolve_generation(fitness_scores)

        # Report
        best = evolver.history[-1]
        print(f"\n  Generation {gen + 1} Summary:")
        print(f"    Best: {best['best_fitness']:.3f}")
        print(f"    Avg:  {best['avg_fitness']:.3f}")
        print(f"    Best prompt: \"{best['best_prompt']}...\"")

    # Final results
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    best = evolver.best_ever
    improvement = ((best.fitness / baseline_fitness) - 1) * 100 if baseline_fitness > 0 else 0

    print(f"\nFitness: {baseline_fitness:.3f} → {best.fitness:.3f} ({improvement:+.1f}%)")
    print()
    print("Best evolved prompt:")
    print("-" * 70)
    print(best.prompt)
    print("-" * 70)
    print()

    # Final evaluation of best
    print("Final evaluation of best prompt:")
    final_negotiator = PersonaNegotiator(persona=best.prompt)

    for scenario in SCENARIOS:
        print(f"\n  {scenario.name}:")
        result = final_negotiator(
            scenario=f"{scenario.name}: {scenario.context}",
            counterpart_position=scenario.counterpart
        )
        print(f"    Reasoning: {result.reasoning[:100]}...")
        print(f"    Response: {result.response[:150]}...")
        scores = evaluate_response(result.response)
        print(f"    Scores: {', '.join(f'{k}={v:.2f}' for k, v in scores.items())}")

    print()
    print("=" * 70)
    print("EVOLUTION HISTORY")
    print("=" * 70)
    for h in evolver.history:
        bar = "█" * int(h['best_fitness'] * 20) + "░" * (20 - int(h['best_fitness'] * 20))
        print(f"  Gen {h['generation']:2d}: [{bar}] {h['best_fitness']:.3f}")

    return evolver


if __name__ == "__main__":
    asyncio.run(run_evolution(generations=8, population_size=10))
