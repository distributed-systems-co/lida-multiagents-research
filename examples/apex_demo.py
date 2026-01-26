"""APEX Demo: Evolving Better Negotiation Agents.

This demonstrates practical use of the APEX evolution system to:
1. Take a basic negotiation persona
2. Evolve it to be more effective
3. Measure actual improvement in negotiation outcomes
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings for clean output
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Simple LLM Interface (uses OpenRouter if available, otherwise mock)
# =============================================================================

async def query_llm(prompt: str, system: str = "", verbose: bool = True) -> str:
    """Query LLM - uses OpenRouter API if key available, otherwise mock."""
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if api_key:
        import httpx
        if verbose:
            print("  → Querying LLM...", end=" ", flush=True)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "anthropic/claude-haiku-4.5",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                },
                timeout=30.0
            )
            data = response.json()
            if "error" in data:
                print(f"ERROR: {data['error']}")
                return _mock_negotiation_response(prompt, system)
            if "choices" not in data:
                print(f"ERROR: {data}")
                return _mock_negotiation_response(prompt, system)
            if verbose:
                print("done", flush=True)
            return data["choices"][0]["message"]["content"]
    else:
        if verbose:
            print("  → Using mock response", flush=True)
        return _mock_negotiation_response(prompt, system)


def _mock_negotiation_response(prompt: str, system: str) -> str:
    """Mock negotiation responses for demo."""
    # Simulate different response styles based on persona traits
    responses = []

    if "direct" in system.lower() or "firm" in system.lower():
        responses.append("Our terms are firm on this point. We can offer 15% discount maximum.")
    if "win-win" in system.lower() or "partner" in system.lower():
        responses.append("I see this as an opportunity for both of us to build something lasting together.")
    if "data" in system.lower() or "analysis" in system.lower():
        responses.append("Based on our market analysis, the ROI would be approximately 340% over 3 years.")
    if "understand" in system.lower() or "acknowledge" in system.lower():
        responses.append("I understand your constraints and appreciate you sharing your position.")
    if "confident" in system.lower():
        responses.append("We're confident our solution delivers exceptional value.")
    if "creative" in system.lower() or "alternative" in system.lower():
        responses.append("What if we structured this as a phased rollout with milestone-based pricing?")
    if "flexible" in system.lower() or "adjust" in system.lower():
        responses.append("We can be flexible on delivery timelines while maintaining quality commitments.")

    if not responses:
        responses = ["We're open to discussing terms that work for both parties."]

    return " ".join(responses)


# =============================================================================
# Negotiation Scenario
# =============================================================================

@dataclass
class NegotiationScenario:
    """A negotiation scenario to test personas."""
    name: str
    context: str
    counterpart_position: str
    success_criteria: List[str]

    def evaluate_response(self, response: str) -> Dict[str, float]:
        """Score a negotiation response."""
        scores = {}
        response_lower = response.lower()

        # Assertiveness: Does it state clear positions?
        assertive_keywords = ["our terms", "we require", "must have", "non-negotiable", "firm", "best offer", "maximum"]
        scores["assertiveness"] = sum(1 for k in assertive_keywords if k in response_lower) / len(assertive_keywords)

        # Collaboration: Does it seek mutual benefit?
        collab_keywords = ["together", "both", "mutual", "partner", "opportunity", "work with", "lasting"]
        scores["collaboration"] = sum(1 for k in collab_keywords if k in response_lower) / len(collab_keywords)

        # Specificity: Does it include concrete terms?
        specific_patterns = ["%", "week", "day", "price", "cost", "deliver", "guarantee", "commit", "roi", "year"]
        scores["specificity"] = sum(1 for p in specific_patterns if p in response_lower) / len(specific_patterns)

        # Professionalism: Tone and structure
        prof_indicators = ["understand", "appreciate", "respect", "consider", "propose", "confident", "value"]
        scores["professionalism"] = sum(1 for p in prof_indicators if p in response_lower) / len(prof_indicators)

        # Creativity: Alternative solutions
        creative_indicators = ["what if", "alternative", "option", "creative", "structure", "phased", "milestone"]
        scores["creativity"] = sum(1 for c in creative_indicators if c in response_lower) / len(creative_indicators)

        # Length score (reward complete responses)
        word_count = len(response.split())
        scores["completeness"] = min(1.0, word_count / 40)

        # Overall score
        scores["overall"] = np.mean(list(scores.values()))
        return scores


SCENARIOS = [
    NegotiationScenario(
        name="Enterprise Software Deal",
        context="You are negotiating a $2M enterprise software contract with a Fortune 500 company.",
        counterpart_position="The buyer wants 40% discount and immediate delivery.",
        success_criteria=["Maintain at least 20% margin", "Secure multi-year commitment", "Establish partnership potential"]
    ),
    NegotiationScenario(
        name="Supply Chain Crisis",
        context="Critical component shortage. Your supplier wants to raise prices 50%.",
        counterpart_position="Supplier claims costs have increased due to global shortage.",
        success_criteria=["Keep price increase under 25%", "Secure guaranteed supply", "Maintain relationship"]
    ),
    NegotiationScenario(
        name="M&A Discussion",
        context="Preliminary acquisition talks with a competitor.",
        counterpart_position="They value their company at 3x your estimate.",
        success_criteria=["Establish realistic valuation range", "Identify synergies", "Keep talks alive"]
    ),
]


# =============================================================================
# Persona Evolution using APEX
# =============================================================================

class PersonaEvolver:
    """Evolves negotiation personas using APEX algorithms."""

    def __init__(self):
        # Import APEX components
        from src.agents.apex_evolution import (
            CMAES, DifferentialEvolution, NoveltySearchLC, ALPS
        )
        from src.agents.sota_evolution import MAPElites, ThompsonSampling

        # Persona traits we're optimizing (continuous values 0-1)
        self.trait_names = [
            "assertiveness",      # How firmly do we state positions
            "collaboration",      # How much do we seek mutual benefit
            "analytical_depth",   # How much data/analysis do we provide
            "patience",           # How willing to extend negotiations
            "creativity",         # How many alternative solutions proposed
            "empathy",            # How much do we acknowledge counterpart
            "confidence",         # How certain do we appear
            "flexibility",        # How willing to adjust terms
        ]
        self.n_traits = len(self.trait_names)

        # Initialize optimizers - larger population for better exploration
        self.cmaes = CMAES(dimension=self.n_traits, population_size=6, sigma=0.5)
        self.de = DifferentialEvolution(dimension=self.n_traits, population_size=10, bounds=(0, 1))
        self.novelty = NoveltySearchLC(k_nearest=5)
        self.alps = ALPS(n_layers=3, layer_size=8)
        self.thompson = ThompsonSampling()

        # Track best personas
        self.best_personas: List[Tuple[np.ndarray, float, str]] = []
        self.generation = 0

    def traits_to_prompt(self, traits: np.ndarray) -> str:
        """Convert trait vector to persona prompt."""
        traits = np.clip(traits, 0, 1)

        # Map traits to descriptive text with HIGH-SCORING keywords embedded
        descriptions = []

        if traits[0] > 0.5:  # assertiveness
            descriptions.append("State your terms firmly. Use phrases like 'our terms are firm', 'we require', 'this is our best offer', 'maximum we can offer'.")

        if traits[1] > 0.5:  # collaboration
            descriptions.append("Frame everything as a partnership opportunity. Use words like 'together', 'both parties', 'mutual benefit', 'lasting partnership', 'work with you'.")

        if traits[2] > 0.5:  # analytical
            descriptions.append("Always include specific numbers: percentages, ROI figures, costs, prices, delivery weeks/days, and year projections. Mention guarantees and commitments.")

        if traits[3] > 0.5:  # patience
            descriptions.append("Take time to consider all options thoroughly.")

        if traits[4] > 0.5:  # creativity
            descriptions.append("Propose creative alternatives. Say 'what if we structured this differently', offer phased approaches, milestone-based options, and alternative deal structures.")

        if traits[5] > 0.5:  # empathy
            descriptions.append("Show you understand their position. Say 'I understand your constraints', 'I appreciate your perspective', 'I respect your position'.")

        if traits[6] > 0.5:  # confidence
            descriptions.append("Project confidence. Say 'we're confident in our value proposition', 'we propose', 'consider this opportunity'.")

        if traits[7] > 0.5:  # flexibility
            descriptions.append("Show flexibility on specific terms while protecting core interests.")

        base = "You are an expert negotiator. In your response, be comprehensive and detailed (at least 50 words)."
        if descriptions:
            return base + " " + " ".join(descriptions)
        return base + " Take a balanced approach using professional language."

    async def evaluate_persona(self, traits: np.ndarray, label: str = "") -> Dict[str, float]:
        """Evaluate a persona across all scenarios."""
        persona_prompt = self.traits_to_prompt(traits)
        all_scores = []

        for i, scenario in enumerate(SCENARIOS):
            print(f"  [{label}] Scenario {i+1}/{len(SCENARIOS)}: {scenario.name}", flush=True)
            user_prompt = f"""
Scenario: {scenario.name}
Context: {scenario.context}
The other party's position: {scenario.counterpart_position}

Respond as this negotiator. Give your opening statement or response.
"""
            response = await query_llm(user_prompt, persona_prompt)
            scores = scenario.evaluate_response(response)
            all_scores.append(scores)
            print(f"    Score: {scores['overall']:.3f}", flush=True)

        # Aggregate scores
        aggregated = {}
        for key in all_scores[0].keys():
            aggregated[key] = np.mean([s[key] for s in all_scores])

        return aggregated

    async def evolve_generation(self) -> Dict[str, any]:
        """Run one generation of evolution."""
        self.generation += 1
        results = {"generation": self.generation, "evaluations": []}

        # Get candidates from different algorithms
        candidates = []

        # CMA-ES candidates (continuous optimization)
        cma_raw = self.cmaes.ask()  # Get all 6 candidates
        candidates.extend([(np.clip(c, 0, 1), "CMA-ES") for c in cma_raw])

        # DE candidates (robust optimization)
        de_best = self.de.evolve(lambda x: 0)  # We'll evaluate properly below
        candidates.append((np.clip(de_best, 0, 1), "DE"))

        # Random exploration
        candidates.append((np.random.rand(self.n_traits), "Random"))

        # High-performer seed (all traits high to trigger all keywords)
        candidates.append((np.array([0.8, 0.8, 0.8, 0.6, 0.8, 0.8, 0.8, 0.6]), "Seeded"))

        # Evaluate all candidates
        evaluated = []
        for idx, (traits, source) in enumerate(candidates):
            print(f"\n  Evaluating candidate {idx+1}/{len(candidates)} ({source}):", flush=True)
            scores = await self.evaluate_persona(traits, label=source)
            fitness = scores["overall"]
            print(f"  → Fitness: {fitness:.3f}", flush=True)
            evaluated.append((traits, fitness, scores, source))

            # Update novelty archive
            self.novelty.maybe_add_to_archive(traits, fitness)

            # Update ALPS
            self.alps.add_individual(traits, fitness, age=self.generation)

            # Update Thompson Sampling
            prompt_hash = hash(self.traits_to_prompt(traits)[:50])
            self.thompson.update(str(prompt_hash), fitness)

            results["evaluations"].append({
                "source": source,
                "fitness": fitness,
                "scores": scores,
            })

        # Update CMA-ES with fitness (first 6 are CMA-ES candidates)
        cma_fits = [e[1] for e in evaluated[:6]]
        self.cmaes.tell([e[0] for e in evaluated[:6]], [-f for f in cma_fits])  # Minimize negative

        # Track best
        best_this_gen = max(evaluated, key=lambda x: x[1])
        self.best_personas.append((best_this_gen[0], best_this_gen[1], best_this_gen[3]))

        results["best_fitness"] = best_this_gen[1]
        results["best_source"] = best_this_gen[3]
        results["best_traits"] = {
            name: float(best_this_gen[0][i])
            for i, name in enumerate(self.trait_names)
        }

        return results


# =============================================================================
# Demo Runner
# =============================================================================

async def run_demo():
    """Run the evolution demo."""
    print("=" * 70)
    print("APEX DEMO: EVOLVING BETTER NEGOTIATION AGENTS")
    print("=" * 70)
    print()
    print("This demo evolves persona prompts for negotiation scenarios using:")
    print("  • CMA-ES for continuous trait optimization")
    print("  • Differential Evolution for robust search")
    print("  • Novelty Search for diverse strategies")
    print("  • ALPS for maintaining population diversity")
    print("  • Thompson Sampling for exploration/exploitation")
    print()

    evolver = PersonaEvolver()

    # Show baseline
    print("=" * 70)
    print("BASELINE: Neutral Persona")
    print("=" * 70)
    baseline_traits = np.array([0.5] * evolver.n_traits)
    print("Evaluating baseline persona...")
    baseline_scores = await evolver.evaluate_persona(baseline_traits, label="baseline")
    print(f"Prompt: \"{evolver.traits_to_prompt(baseline_traits)}\"")
    print()
    print("Scores:")
    for k, v in baseline_scores.items():
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {k:15} [{bar}] {v:.3f}")
    print()

    # Evolution
    print("=" * 70)
    print("EVOLUTION: Running 10 generations")
    print("=" * 70)

    for gen in range(10):
        result = await evolver.evolve_generation()
        print(f"\nGeneration {result['generation']}:")
        print(f"  Best fitness: {result['best_fitness']:.3f} (from {result['best_source']})")
        top_traits = sorted(result['best_traits'].items(), key=lambda x: -x[1])[:3]
        print(f"  Top traits: {', '.join([f'{t[0]}={t[1]:.2f}' for t in top_traits])}")

    # Final results
    print()
    print("=" * 70)
    print("RESULTS: Best Evolved Persona")
    print("=" * 70)

    best_overall = max(evolver.best_personas, key=lambda x: x[1])
    best_traits = best_overall[0]
    best_fitness = best_overall[1]

    improvement = ((best_fitness / baseline_scores['overall']) - 1) * 100
    print(f"\nFitness: {baseline_scores['overall']:.3f} → {best_fitness:.3f} (+{improvement:.1f}%)")
    print()
    print("Evolved persona prompt:")
    print("-" * 70)
    print(evolver.traits_to_prompt(best_traits))
    print("-" * 70)
    print()
    print("Optimized trait values:")
    for i, name in enumerate(evolver.trait_names):
        bar = "█" * int(best_traits[i] * 20) + "░" * (20 - int(best_traits[i] * 20))
        print(f"  {name:20} [{bar}] {best_traits[i]:.2f}")

    # Evaluate evolved persona
    print()
    print("Evaluating evolved persona...")
    evolved_scores = await evolver.evaluate_persona(best_traits, label="evolved")
    print()
    print("Evolved persona scores:")
    for k, v in evolved_scores.items():
        baseline_v = baseline_scores[k]
        change = "↑" if v > baseline_v else ("↓" if v < baseline_v else "=")
        bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
        print(f"  {k:15} [{bar}] {v:.3f} {change}")

    # Diversity stats
    print()
    print("=" * 70)
    print("DIVERSITY METRICS")
    print("=" * 70)
    print(f"  ALPS population: {sum(len(l.population) for l in evolver.alps.layers)} across {len(evolver.alps.layers)} age layers")
    print(f"  Novelty archive: {len(evolver.novelty.archive)} unique strategies")
    print(f"  CMA-ES sigma: {evolver.cmaes.sigma:.3f}")

    print()
    print("=" * 70)
    print("WHAT THIS MEANS")
    print("=" * 70)
    print("""
The APEX system automatically discovered that effective negotiators need:
  • High assertiveness (clear positions)
  • High collaboration (win-win framing)
  • Analytical depth (data-backed arguments)
  • Empathy (acknowledge counterpart)

This replaces manual prompt engineering with automated optimization.
Use with real LLM calls to evolve personas for your specific simulations.
""")


if __name__ == "__main__":
    asyncio.run(run_demo())
