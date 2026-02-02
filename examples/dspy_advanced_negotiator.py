"""Advanced DSPy Neuroevolved Negotiation System.

Combines multiple DSPy optimization strategies:
- GEPA: Reflective prompt evolution with Pareto selection
- MIPROv2: Instruction + few-shot co-optimization via Bayesian search
- ReAct: Tool-augmented reasoning for market research
- LLM-as-Judge: Rich semantic evaluation beyond keyword matching

This demonstrates state-of-the-art prompt optimization for complex tasks.
"""

import os
import sys
import json
from typing import Optional, Union, List, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from dspy import (
    Example, Prediction, Module, ChainOfThought, ReAct,
    Signature, InputField, OutputField, Tool
)

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for the optimization run."""
    main_model: str = "openrouter/anthropic/claude-haiku-4.5"
    reflection_model: str = "openrouter/anthropic/claude-sonnet-4"
    judge_model: str = "openrouter/anthropic/claude-sonnet-4"
    optimizer: str = "gepa"  # gepa, mipro, or compare
    budget: str = "medium"   # light, medium, heavy
    num_threads: int = 2
    seed: int = 42


def setup_dspy(config: Config):
    """Configure DSPy with multiple LMs."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY required")

    main_lm = dspy.LM(
        model=config.main_model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=2500,
    )
    dspy.configure(lm=main_lm)

    reflection_lm = dspy.LM(
        model=config.reflection_model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=4000,
        temperature=1.0,
    )

    judge_lm = dspy.LM(
        model=config.judge_model,
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=2000,
        temperature=0.0,
    )

    return main_lm, reflection_lm, judge_lm


# =============================================================================
# Tools for ReAct-based Negotiation Research
# =============================================================================

def market_research(query: str) -> str:
    """Research market conditions, competitor pricing, and industry benchmarks.

    Args:
        query: What to research (e.g., "enterprise software pricing", "SaaS discount norms")

    Returns:
        Market research findings relevant to the negotiation.
    """
    # Simulated market data - in production, this would call real APIs
    research_data = {
        "enterprise software": "Enterprise software deals typically range from $50-500 per user/month. Standard discounts for Fortune 500: 15-25%. Multi-year deals can justify 30-40% discounts. Implementation typically 8-12 weeks.",
        "saas pricing": "SaaS industry averages: 20% discount for annual prepay, 10-15% for 3-year commits. Startup discounts (YC, etc.) range 25-50%. Enterprise tiers average 3x SMB pricing.",
        "manufacturing": "Manufacturing component margins: 15-40% typical. Supply chain disruptions justify 10-20% increases. Volume discounts: 5% per 100% volume increase.",
        "consulting rates": "Big 4 consulting: $300-600/hr. Boutique firms: $200-400/hr. Independent consultants: $150-300/hr. Value-based pricing can be 2-3x hourly equivalent.",
        "m&a valuations": "Software M&A: 3-8x revenue for growth companies, 10-20x EBITDA for mature. Strategic premium: 20-40% over market. Earnouts typical for 20-40% of deal value.",
    }

    for key, data in research_data.items():
        if key in query.lower():
            return data

    return "Industry benchmarks suggest standard negotiation ranges of 10-30% flexibility on pricing, with larger discounts tied to volume, commitment length, or strategic value."


def competitor_analysis(company_or_product: str) -> str:
    """Analyze competitor positioning and pricing.

    Args:
        company_or_product: The competitor or product category to analyze

    Returns:
        Competitive intelligence for negotiation leverage.
    """
    analyses = {
        "salesforce": "Salesforce: $25-300/user/month. Known for aggressive discounting (30-50%) at quarter end. Lock-in concerns. Implementation complexity.",
        "aws": "AWS: Pay-as-you-go with committed use discounts up to 72%. Competitors (Azure, GCP) often match or beat pricing. Enterprise agreements negotiable.",
        "oracle": "Oracle: License + 22% annual support. Known for audit pressure. Alternatives: PostgreSQL, cloud-native options. Negotiate support reduction.",
        "microsoft": "Microsoft: EA agreements with 3-year commits. Education/nonprofit: 50-85% discounts. Compete with Google Workspace for leverage.",
    }

    for key, analysis in analyses.items():
        if key in company_or_product.lower():
            return analysis

    return "Competitor analysis suggests multiple alternatives exist. Use competitive pressure for 15-25% better terms. Request proof of differentiation."


def calculate_roi(scenario: str) -> str:
    """Calculate ROI and financial metrics for the deal.

    Args:
        scenario: Description of the deal scenario

    Returns:
        ROI analysis and financial justification.
    """
    return """ROI Analysis:
- Typical enterprise software ROI: 200-400% over 3 years
- Payback period benchmark: 12-18 months
- TCO considerations: License (40%), implementation (25%), training (15%), ongoing support (20%)
- Productivity gains: 15-30% efficiency improvement typical
- Risk-adjusted NPV should exceed 150% of investment for approval"""


# =============================================================================
# Signatures for Multi-Stage Negotiation
# =============================================================================

class ResearchSignature(Signature):
    """Research the negotiation context to inform strategy."""

    scenario: str = InputField(desc="The negotiation scenario and context")
    counterpart_position: str = InputField(desc="The other party's stated position")

    market_context: str = OutputField(desc="Relevant market data and benchmarks")
    leverage_points: str = OutputField(desc="Key points of leverage in this negotiation")
    risks: str = OutputField(desc="Risks and potential objections to anticipate")


class StrategySignature(Signature):
    """Develop a negotiation strategy based on research."""

    scenario: str = InputField(desc="The negotiation scenario")
    counterpart_position: str = InputField(desc="The other party's position")
    research: str = InputField(desc="Market research and context")

    opening_position: str = OutputField(desc="Your ideal opening position")
    batna: str = OutputField(desc="Your BATNA (Best Alternative To Negotiated Agreement)")
    concession_strategy: str = OutputField(desc="What you can concede and in what order")
    key_arguments: str = OutputField(desc="Main arguments to support your position")


class ResponseSignature(Signature):
    """Generate the actual negotiation response."""

    scenario: str = InputField(desc="The negotiation scenario")
    counterpart_position: str = InputField(desc="What they're demanding")
    strategy: str = InputField(desc="Your negotiation strategy")

    response: str = OutputField(desc="Your negotiation response (be specific, professional, and strategic)")


class JudgeSignature(Signature):
    """Evaluate a negotiation response as an expert negotiation coach."""

    scenario: str = InputField(desc="The negotiation scenario")
    counterpart_position: str = InputField(desc="The counterpart's position")
    response: str = InputField(desc="The negotiation response to evaluate")

    assertiveness_score: float = OutputField(desc="Score 0-1: Does it state clear positions?")
    collaboration_score: float = OutputField(desc="Score 0-1: Does it seek mutual benefit?")
    specificity_score: float = OutputField(desc="Score 0-1: Are terms concrete and specific?")
    professionalism_score: float = OutputField(desc="Score 0-1: Is the tone appropriate?")
    creativity_score: float = OutputField(desc="Score 0-1: Does it offer alternatives?")
    strategic_score: float = OutputField(desc="Score 0-1: Is the strategy sound?")
    critique: str = OutputField(desc="Detailed feedback on how to improve")
    overall_score: float = OutputField(desc="Overall score 0-1")


# =============================================================================
# DSPy Modules
# =============================================================================

class SimpleNegotiator(Module):
    """Basic single-stage negotiator."""

    def __init__(self):
        super().__init__()
        self.respond = ChainOfThought(ResponseSignature)

    def forward(self, scenario: str, counterpart_position: str) -> Prediction:
        # Create a basic strategy inline
        strategy = "Maintain firm position while showing flexibility on secondary terms."
        return self.respond(
            scenario=scenario,
            counterpart_position=counterpart_position,
            strategy=strategy
        )


class ResearchNegotiator(Module):
    """Two-stage negotiator with research phase."""

    def __init__(self):
        super().__init__()
        self.research = ChainOfThought(ResearchSignature)
        self.strategize = ChainOfThought(StrategySignature)
        self.respond = ChainOfThought(ResponseSignature)

    def forward(self, scenario: str, counterpart_position: str) -> Prediction:
        # Research phase
        research_result = self.research(
            scenario=scenario,
            counterpart_position=counterpart_position
        )

        # Strategy phase
        research_summary = f"""
Market Context: {research_result.market_context}
Leverage Points: {research_result.leverage_points}
Risks: {research_result.risks}
"""
        strategy_result = self.strategize(
            scenario=scenario,
            counterpart_position=counterpart_position,
            research=research_summary
        )

        # Response phase
        strategy_summary = f"""
Opening: {strategy_result.opening_position}
BATNA: {strategy_result.batna}
Concessions: {strategy_result.concession_strategy}
Arguments: {strategy_result.key_arguments}
"""
        return self.respond(
            scenario=scenario,
            counterpart_position=counterpart_position,
            strategy=strategy_summary
        )


class ToolResearchSignature(Signature):
    """Research and respond to a negotiation scenario using available tools."""
    scenario: str = InputField(desc="The negotiation scenario")
    counterpart_position: str = InputField(desc="The counterpart's position")
    response: str = OutputField(desc="Your negotiation response")


class ToolAugmentedNegotiator(Module):
    """ReAct-based negotiator with tool use for research."""

    def __init__(self):
        super().__init__()

        # Define tools
        tools = [
            Tool(
                func=market_research,
                name="market_research",
                desc="Research market conditions, pricing benchmarks, and industry standards",
                args={"query": {"type": "string", "desc": "What to research"}}
            ),
            Tool(
                func=competitor_analysis,
                name="competitor_analysis",
                desc="Analyze competitor positioning and pricing for leverage",
                args={"company_or_product": {"type": "string", "desc": "Competitor to analyze"}}
            ),
            Tool(
                func=calculate_roi,
                name="calculate_roi",
                desc="Calculate ROI and financial justification for the deal",
                args={"scenario": {"type": "string", "desc": "Deal scenario"}}
            ),
        ]

        self.react = ReAct(
            signature=ToolResearchSignature,
            tools=tools,
            max_iters=5,
        )

    def forward(self, scenario: str, counterpart_position: str) -> Prediction:
        return self.react(scenario=scenario, counterpart_position=counterpart_position)


# =============================================================================
# Dataset
# =============================================================================

def create_dataset():
    """Create training and validation examples."""

    examples = [
        Example(
            scenario="You represent a software vendor negotiating a $2M enterprise contract with a Fortune 500 company. Minimum acceptable margin: 20%.",
            counterpart_position="Buyer demands 40% discount and 2-week delivery instead of standard 8 weeks.",
            ideal_outcome="15-25% discount with phased delivery, multi-year commitment secured",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Selling cloud infrastructure to a fast-growing startup. They could become a major account in 2-3 years.",
            counterpart_position="They want 60% startup discount plus unlimited support because 'we're early stage'.",
            ideal_outcome="Growth-based pricing with success milestones, capped support hours",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Your SaaS product is being compared to a cheaper competitor. Prospect is price-sensitive.",
            counterpart_position="Competitor offers similar features at half price. Match it or we walk.",
            ideal_outcome="Differentiation on TCO/ROI, value-add services, or tiered pricing",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Procurement lead negotiating with supplier raising prices 50% due to shortages.",
            counterpart_position="Costs up 60%, other buyers waiting. Take it or leave it.",
            ideal_outcome="10-25% increase max, volume commitment, phased increases",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Renegotiating manufacturing contract due to quality issues.",
            counterpart_position="Manufacturer blames your specs, wants price increase to 'improve quality'.",
            ideal_outcome="Quality SLAs with penalties, no price increase, clear metrics",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Acquisition talks with competitor. Your budget: $150-180M.",
            counterpart_position="They want $500M citing 'strategic value' and future potential.",
            ideal_outcome="$150-200M with earnouts tied to performance milestones",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Negotiating strategic partnership for exclusive technology access.",
            counterpart_position="They want worldwide 5-year exclusivity for $5M upfront.",
            ideal_outcome="Limited exclusivity, performance requirements, growth sharing",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Consultant negotiating major engagement with budget-constrained client.",
            counterpart_position="Love the proposal but can only pay 50% of quoted rate.",
            ideal_outcome="Scope adjustment, success fees, or phased engagement",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Negotiating office lease renewal in a tenant's market.",
            counterpart_position="Landlord wants 15% increase despite 20% vacancy in building.",
            ideal_outcome="Flat or reduced rent, improvement allowance, flexible terms",
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="Licensing your patent portfolio to a large manufacturer.",
            counterpart_position="They offer flat $1M for perpetual worldwide license.",
            ideal_outcome="Per-unit royalty or higher upfront with volume tiers",
        ).with_inputs("scenario", "counterpart_position"),
    ]

    trainset = examples[:7]
    valset = examples[7:]

    return trainset, valset


# =============================================================================
# LLM-as-Judge Metric
# =============================================================================

class NegotiationJudge:
    """LLM-based evaluation of negotiation responses."""

    def __init__(self, judge_lm):
        self.judge_lm = judge_lm
        self.judge = ChainOfThought(JudgeSignature)

    def evaluate(self, scenario: str, counterpart_position: str, response: str) -> dict:
        """Get detailed LLM evaluation of a response."""
        with dspy.context(lm=self.judge_lm):
            result = self.judge(
                scenario=scenario,
                counterpart_position=counterpart_position,
                response=response
            )

        return {
            'assertiveness': float(result.assertiveness_score),
            'collaboration': float(result.collaboration_score),
            'specificity': float(result.specificity_score),
            'professionalism': float(result.professionalism_score),
            'creativity': float(result.creativity_score),
            'strategic': float(result.strategic_score),
            'overall': float(result.overall_score),
            'critique': result.critique,
        }


def create_gepa_metric(judge: NegotiationJudge):
    """Create a GEPA-compatible metric.

    Returns float for normal evaluation, dict for GEPA reflection when trace is provided.
    """

    def metric(
        gold: Example,
        pred: Prediction,
        trace=None,
        pred_name: Optional[str] = None,
        pred_trace=None,
    ) -> Union[float, dict]:
        response = getattr(pred, 'response', '') or ''

        if not response or len(response) < 20:
            score = 0.0
            feedback = "Response is empty or too short."
            if trace is not None:
                return {"score": score, "feedback": feedback}
            return score

        try:
            eval_result = judge.evaluate(
                scenario=gold.scenario,
                counterpart_position=gold.counterpart_position,
                response=response
            )
            score = float(eval_result['overall'])
            feedback = f"""Score: {score:.2f}
Assertiveness: {eval_result['assertiveness']:.2f}, Collaboration: {eval_result['collaboration']:.2f}
Specificity: {eval_result['specificity']:.2f}, Professionalism: {eval_result['professionalism']:.2f}
Creativity: {eval_result['creativity']:.2f}, Strategic: {eval_result['strategic']:.2f}
Critique: {eval_result['critique']}"""

        except Exception:
            # Fallback to keyword-based scoring
            r = response.lower()
            scores = []
            for markers in [
                ["our position", "we require", "firm", "terms"],
                ["together", "mutual", "partnership", "both"],
                ["%", "roi", "cost", "week", "month"],
                ["understand", "appreciate", "confident"],
                ["alternative", "option", "what if", "structure"],
            ]:
                scores.append(min(1.0, sum(1 for m in markers if m in r) / 2))
            scores.append(min(1.0, len(response.split()) / 80))
            score = sum(scores) / len(scores)
            feedback = f"Keyword-based score: {score:.2f}"

        # GEPA reflection needs dict, normal eval needs float
        if trace is not None:
            return {"score": score, "feedback": feedback}
        return score

    return metric


def create_mipro_metric(judge: NegotiationJudge):
    """Create a MIPROv2-compatible metric (returns just score)."""

    def metric(gold: Example, pred: Prediction, trace=None) -> float:
        response = getattr(pred, 'response', '') or ''

        if not response or len(response) < 20:
            return 0.0

        try:
            eval_result = judge.evaluate(
                scenario=gold.scenario,
                counterpart_position=gold.counterpart_position,
                response=response
            )
            return eval_result['overall']

        except Exception:
            # Fallback
            r = response.lower()
            markers = ["our position", "together", "mutual", "%", "understand", "alternative"]
            return min(1.0, sum(1 for m in markers if m in r) / 3)

    return metric


# =============================================================================
# Main Optimization
# =============================================================================

def run_optimization(config: Config):
    """Run the advanced negotiation optimization."""

    print("=" * 70)
    print("ADVANCED DSPy NEGOTIATION OPTIMIZATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Main model:       {config.main_model}")
    print(f"  Reflection model: {config.reflection_model}")
    print(f"  Judge model:      {config.judge_model}")
    print(f"  Optimizer:        {config.optimizer}")
    print(f"  Budget:           {config.budget}")
    print()

    # Setup
    print("Initializing DSPy...", flush=True)
    main_lm, reflection_lm, judge_lm = setup_dspy(config)
    judge = NegotiationJudge(judge_lm)
    print("Done.\n")

    # Dataset
    print("Creating dataset...", flush=True)
    trainset, valset = create_dataset()
    print(f"  Train: {len(trainset)}, Val: {len(valset)}\n")

    # Choose student module
    print("Creating negotiator module...", flush=True)
    student = ResearchNegotiator()  # Multi-stage with research
    print("  Using: ResearchNegotiator (3-stage: research → strategy → response)\n")

    # Baseline evaluation
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)

    baseline_scores = []
    for i, ex in enumerate(valset):
        print(f"\n  [{i+1}/{len(valset)}] {ex.scenario[:50]}...")
        try:
            pred = student(scenario=ex.scenario, counterpart_position=ex.counterpart_position)
            result = judge.evaluate(ex.scenario, ex.counterpart_position, pred.response)
            baseline_scores.append(result['overall'])
            print(f"    Score: {result['overall']:.3f}")
            print(f"    Response: {pred.response[:80]}...")
        except Exception as e:
            print(f"    Error: {e}")
            baseline_scores.append(0.0)

    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    print(f"\n  Baseline average: {baseline_avg:.3f}\n")

    # Optimization
    print("=" * 70)
    print(f"OPTIMIZATION ({config.optimizer.upper()})")
    print("=" * 70)
    print()

    if config.optimizer == "gepa":
        gepa_metric = create_gepa_metric(judge)

        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto=config.budget,
            reflection_lm=reflection_lm,
            reflection_minibatch_size=3,
            candidate_selection_strategy="pareto",
            use_merge=True,
            max_merge_invocations=5,
            track_stats=True,
            track_best_outputs=True,
            num_threads=config.num_threads,
            seed=config.seed,
            add_format_failure_as_feedback=True,
        )

        print("Starting GEPA optimization...")
        print("  • Pareto-based candidate selection")
        print("  • LLM reflection on execution traces")
        print("  • Rich textual feedback guiding mutations")
        print("  • Merge operation for combining variants")
        print()

        optimized = optimizer.compile(
            student=student,
            trainset=trainset,
            valset=valset,
        )

    elif config.optimizer == "mipro":
        mipro_metric = create_mipro_metric(judge)

        optimizer = dspy.MIPROv2(
            metric=mipro_metric,
            auto=config.budget,
            num_threads=config.num_threads,
            seed=config.seed,
            track_stats=True,
            verbose=True,
        )

        print("Starting MIPROv2 optimization...")
        print("  • Bayesian optimization over instructions")
        print("  • Few-shot example bootstrapping")
        print("  • Program-aware instruction proposal")
        print()

        optimized = optimizer.compile(
            student=student,
            trainset=trainset,
            valset=valset,
        )

    elif config.optimizer == "compare":
        print("Running both optimizers for comparison...")

        # GEPA
        print("\n--- GEPA ---")
        gepa_metric = create_gepa_metric(judge)
        gepa_optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto="light",
            reflection_lm=reflection_lm,
            track_stats=True,
            seed=config.seed,
        )
        gepa_optimized = gepa_optimizer.compile(student=student, trainset=trainset, valset=valset)

        # MIPROv2
        print("\n--- MIPROv2 ---")
        mipro_metric = create_mipro_metric(judge)
        mipro_optimizer = dspy.MIPROv2(
            metric=mipro_metric,
            auto="light",
            seed=config.seed,
        )
        mipro_student = ResearchNegotiator()  # Fresh copy
        mipro_optimized = mipro_optimizer.compile(student=mipro_student, trainset=trainset, valset=valset)

        # Compare
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        for name, opt_model in [("GEPA", gepa_optimized), ("MIPROv2", mipro_optimized)]:
            scores = []
            for ex in valset:
                try:
                    pred = opt_model(scenario=ex.scenario, counterpart_position=ex.counterpart_position)
                    result = judge.evaluate(ex.scenario, ex.counterpart_position, pred.response)
                    scores.append(result['overall'])
                except:
                    scores.append(0.0)
            avg = sum(scores) / len(scores) if scores else 0
            print(f"  {name}: {avg:.3f}")

        optimized = gepa_optimized  # Return GEPA for further analysis

    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Final evaluation
    print()
    print("=" * 70)
    print("OPTIMIZED EVALUATION")
    print("=" * 70)

    optimized_scores = []
    for i, ex in enumerate(valset):
        print(f"\n  [{i+1}/{len(valset)}] {ex.scenario[:50]}...")
        try:
            pred = optimized(scenario=ex.scenario, counterpart_position=ex.counterpart_position)
            result = judge.evaluate(ex.scenario, ex.counterpart_position, pred.response)
            optimized_scores.append(result['overall'])
            print(f"    Score: {result['overall']:.3f}")
            print(f"    Response: {pred.response[:80]}...")
        except Exception as e:
            print(f"    Error: {e}")
            optimized_scores.append(0.0)

    optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
    improvement = ((optimized_avg / baseline_avg) - 1) * 100 if baseline_avg > 0 else 0

    # Summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline:    {baseline_avg:.3f}")
    print(f"  Optimized:   {optimized_avg:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")

    # Show evolved components
    print()
    print("=" * 70)
    print("EVOLVED COMPONENTS")
    print("=" * 70)

    for name, predictor in optimized.named_predictors():
        print(f"\n  Predictor: {name}")
        if hasattr(predictor, 'signature') and predictor.signature.__doc__:
            doc = predictor.signature.__doc__
            if len(doc) > 200:
                doc = doc[:200] + "..."
            print(f"    Instruction: {doc}")

    # Statistics
    if hasattr(optimized, 'detailed_results') and optimized.detailed_results:
        results = optimized.detailed_results
        print()
        print("=" * 70)
        print("OPTIMIZATION STATISTICS")
        print("=" * 70)
        print(f"  Candidates explored: {len(results.candidates)}")
        if results.total_metric_calls:
            print(f"  Total metric calls: {results.total_metric_calls}")
        print(f"  Best candidate: #{results.best_idx}")
        print(f"  Pareto scores: {[f'{s:.3f}' for s in results.val_aggregate_scores[:5]]}...")

    return optimized, baseline_avg, optimized_avg


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Advanced DSPy Negotiation Optimizer")
    parser.add_argument("--optimizer", choices=["gepa", "mipro", "compare"], default="gepa")
    parser.add_argument("--budget", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(
        optimizer=args.optimizer,
        budget=args.budget,
        seed=args.seed,
    )

    run_optimization(config)
