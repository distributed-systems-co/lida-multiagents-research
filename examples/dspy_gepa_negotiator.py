"""DSPy GEPA: Reflective Prompt Evolution for Negotiation.

Uses DSPy's GEPA optimizer (Genetic-Pareto) for actual neuroevolved prompts.
GEPA uses LLM reflection on execution traces to propose better instructions.
"""

import os
import sys
from typing import Optional, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dspy
from dspy import Example, Prediction, Module, ChainOfThought, Signature, InputField, OutputField

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

    # Main LM for the negotiator
    lm = dspy.LM(
        model="openrouter/anthropic/claude-3.5-haiku",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=800,
    )
    dspy.configure(lm=lm)

    # Reflection LM (stronger model for GEPA's reflective mutations)
    reflection_lm = dspy.LM(
        model="openrouter/anthropic/claude-sonnet-4",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        max_tokens=4000,
        temperature=1.0,  # GEPA recommends temperature=1 for reflection
    )

    return lm, reflection_lm


# =============================================================================
# Negotiation Signature and Module
# =============================================================================

class NegotiationSignature(Signature):
    """Generate a strategic negotiation response.

    You are negotiating on behalf of your organization. Analyze the situation,
    consider both parties' interests, and craft a response that advances your
    position while maintaining the possibility of a deal.
    """

    scenario: str = InputField(desc="The negotiation context and your role")
    counterpart_position: str = InputField(desc="What the other party is demanding or proposing")

    strategy: str = OutputField(desc="Your negotiation strategy and key points to make")
    response: str = OutputField(desc="Your actual negotiation response to deliver")


class Negotiator(Module):
    """DSPy module for negotiation responses."""

    def __init__(self):
        super().__init__()
        self.negotiate = ChainOfThought(NegotiationSignature)

    def forward(self, scenario: str, counterpart_position: str) -> Prediction:
        return self.negotiate(scenario=scenario, counterpart_position=counterpart_position)


# =============================================================================
# Training/Validation Data
# =============================================================================

def create_negotiation_dataset():
    """Create training and validation examples."""

    examples = [
        # Enterprise deals
        Example(
            scenario="You represent a software vendor negotiating a $2M enterprise contract with a Fortune 500 company. Your minimum acceptable margin is 20%.",
            counterpart_position="The buyer demands 40% discount and delivery within 2 weeks instead of the standard 8 weeks.",
            ideal_elements=["margin protection", "timeline flexibility", "value proposition", "alternative structures"]
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="You're selling cloud infrastructure services. The client is a fast-growing startup that could become a major account.",
            counterpart_position="They want a 60% discount because they're a startup, plus unlimited support.",
            ideal_elements=["growth potential framing", "tiered pricing", "success milestones", "partnership language"]
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="You represent a SaaS company. The prospect is comparing you to a cheaper competitor.",
            counterpart_position="Your competitor offers similar features at half the price. Match their price or we walk.",
            ideal_elements=["differentiation", "TCO analysis", "ROI metrics", "risk of cheaper option"]
        ).with_inputs("scenario", "counterpart_position"),

        # Supply chain
        Example(
            scenario="You're the procurement lead. Your critical component supplier wants to raise prices 50% citing global shortages.",
            counterpart_position="The supplier says costs increased 60% and they have other buyers waiting. Take it or leave it.",
            ideal_elements=["relationship leverage", "volume commitments", "alternative suppliers", "phased increases"]
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="You need to renegotiate a manufacturing contract. Quality has been inconsistent.",
            counterpart_position="The manufacturer blames your specifications and wants to increase prices to improve quality.",
            ideal_elements=["quality metrics", "accountability", "improvement milestones", "penalty clauses"]
        ).with_inputs("scenario", "counterpart_position"),

        # M&A / Partnership
        Example(
            scenario="Preliminary acquisition talks with a competitor. You're the potential acquirer with a budget of $150-180M.",
            counterpart_position="They value their company at $500M based on 'strategic value' and future potential.",
            ideal_elements=["valuation methodology", "synergy analysis", "earnout structures", "due diligence"]
        ).with_inputs("scenario", "counterpart_position"),

        Example(
            scenario="You're negotiating a strategic partnership with a larger company that wants exclusive access to your technology.",
            counterpart_position="They want worldwide exclusivity for 5 years in exchange for a $5M upfront payment.",
            ideal_elements=["exclusivity limitations", "performance requirements", "termination rights", "growth sharing"]
        ).with_inputs("scenario", "counterpart_position"),

        # Employment / Consulting
        Example(
            scenario="You're a consultant negotiating a major engagement. The client has budget constraints.",
            counterpart_position="They love your proposal but can only pay 50% of your quoted rate.",
            ideal_elements=["scope adjustment", "value-based pricing", "success fees", "reference value"]
        ).with_inputs("scenario", "counterpart_position"),
    ]

    # Split into train/val
    trainset = examples[:6]
    valset = examples[6:]

    return trainset, valset


# =============================================================================
# GEPA Feedback Metric
# =============================================================================

def negotiation_metric(
    gold: Example,
    pred: Prediction,
    trace=None,
    pred_name: Optional[str] = None,
    pred_trace=None,
) -> Union[float, dict]:
    """
    GEPA feedback metric for negotiation quality.

    Returns score + textual feedback to guide reflective evolution.
    """
    response = getattr(pred, 'response', '') or ''
    strategy = getattr(pred, 'strategy', '') or ''
    full_output = f"{strategy} {response}".lower()

    # Score components
    scores = {}
    feedback_parts = []

    # 1. Assertiveness (states clear positions)
    assertive_markers = ["our position", "we require", "our terms", "firm on", "non-negotiable",
                        "minimum", "maximum", "best offer", "final offer"]
    assertive_count = sum(1 for m in assertive_markers if m in full_output)
    scores['assertiveness'] = min(1.0, assertive_count / 3)
    if scores['assertiveness'] < 0.5:
        feedback_parts.append("Response lacks assertiveness - should state clear positions using phrases like 'our terms', 'we require', 'our position is firm'")

    # 2. Collaboration (seeks mutual benefit)
    collab_markers = ["together", "mutual", "both parties", "win-win", "partnership",
                     "long-term", "relationship", "work with you", "collaborate"]
    collab_count = sum(1 for m in collab_markers if m in full_output)
    scores['collaboration'] = min(1.0, collab_count / 3)
    if scores['collaboration'] < 0.5:
        feedback_parts.append("Response doesn't frame as collaborative - use 'mutual benefit', 'partnership', 'together', 'both parties'")

    # 3. Specificity (concrete terms, numbers)
    specific_markers = ["%", "roi", "cost", "price", "$", "week", "month", "day", "year",
                       "deliver", "guarantee", "commit", "milestone", "metric"]
    specific_count = sum(1 for m in specific_markers if m in full_output)
    scores['specificity'] = min(1.0, specific_count / 4)
    if scores['specificity'] < 0.5:
        feedback_parts.append("Response lacks specificity - include concrete numbers, percentages, timelines, costs, ROI figures")

    # 4. Professionalism (tone)
    prof_markers = ["understand", "appreciate", "respect", "acknowledge", "recognize",
                   "confident", "pleased", "value", "consider", "propose"]
    prof_count = sum(1 for m in prof_markers if m in full_output)
    scores['professionalism'] = min(1.0, prof_count / 3)
    if scores['professionalism'] < 0.5:
        feedback_parts.append("Response could be more professional - use 'I understand', 'I appreciate', 'we're confident', 'we propose'")

    # 5. Creativity (alternative solutions)
    creative_markers = ["what if", "alternative", "option", "structure", "phased",
                       "milestone", "creative", "flexible", "consider", "explore"]
    creative_count = sum(1 for m in creative_markers if m in full_output)
    scores['creativity'] = min(1.0, creative_count / 3)
    if scores['creativity'] < 0.5:
        feedback_parts.append("Response doesn't offer alternatives - propose creative structures, phased approaches, 'what if' options")

    # 6. Completeness
    word_count = len(response.split())
    scores['completeness'] = min(1.0, word_count / 80)
    if scores['completeness'] < 0.7:
        feedback_parts.append(f"Response too brief ({word_count} words) - provide more comprehensive response (aim for 80+ words)")

    # 7. Ideal elements coverage (from gold)
    ideal_elements = getattr(gold, 'ideal_elements', [])
    if ideal_elements:
        covered = sum(1 for elem in ideal_elements if any(word in full_output for word in elem.lower().split()))
        scores['ideal_coverage'] = covered / len(ideal_elements)
        missing = [e for e in ideal_elements if not any(word in full_output for word in e.lower().split())]
        if missing:
            feedback_parts.append(f"Missing key elements: {', '.join(missing)}")

    # Overall score
    overall = sum(scores.values()) / len(scores)

    # Build feedback string
    if overall >= 0.8:
        feedback = f"Strong negotiation response (score: {overall:.2f}). Minor improvements: " + "; ".join(feedback_parts[:2]) if feedback_parts else "Excellent response."
    elif overall >= 0.5:
        feedback = f"Moderate response (score: {overall:.2f}). Key improvements needed: " + "; ".join(feedback_parts[:3])
    else:
        feedback = f"Weak response (score: {overall:.2f}). Critical issues: " + "; ".join(feedback_parts)

    # Add score breakdown
    feedback += f"\n\nScore breakdown: {', '.join(f'{k}={v:.2f}' for k, v in scores.items())}"

    return {"score": overall, "feedback": feedback}


# =============================================================================
# Main GEPA Optimization
# =============================================================================

def run_gepa_optimization():
    """Run GEPA optimization on the negotiator."""

    print("=" * 70)
    print("DSPy GEPA: REFLECTIVE PROMPT EVOLUTION FOR NEGOTIATION")
    print("=" * 70)
    print()
    print("GEPA (Genetic-Pareto) optimizer features:")
    print("  • LLM reflection on execution traces to propose mutations")
    print("  • Pareto frontier for multi-objective optimization")
    print("  • Rich textual feedback guides prompt evolution")
    print("  • Merge operation combines successful variants")
    print()

    # Setup
    print("Setting up DSPy...", flush=True)
    lm, reflection_lm = setup_dspy()
    print(f"  Main LM: claude-3.5-haiku")
    print(f"  Reflection LM: claude-sonnet-4")
    print()

    # Create dataset
    print("Creating negotiation dataset...", flush=True)
    trainset, valset = create_negotiation_dataset()
    print(f"  Training examples: {len(trainset)}")
    print(f"  Validation examples: {len(valset)}")
    print()

    # Create student module
    student = Negotiator()

    # Evaluate baseline
    print("=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)

    baseline_scores = []
    for i, example in enumerate(valset):
        print(f"\n  Example {i+1}: {example.scenario[:60]}...")
        try:
            pred = student(scenario=example.scenario, counterpart_position=example.counterpart_position)
            result = negotiation_metric(example, pred)
            score = result['score'] if isinstance(result, dict) else result
            baseline_scores.append(score)
            print(f"    Score: {score:.3f}")
            print(f"    Response preview: {pred.response[:100]}...")
        except Exception as e:
            print(f"    Error: {e}")
            baseline_scores.append(0.0)

    baseline_avg = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0
    print(f"\n  Baseline average: {baseline_avg:.3f}")
    print()

    # Run GEPA optimization
    print("=" * 70)
    print("GEPA OPTIMIZATION")
    print("=" * 70)
    print()

    gepa = dspy.GEPA(
        metric=negotiation_metric,
        auto="medium",  # Budget: light/medium/heavy
        reflection_lm=reflection_lm,
        reflection_minibatch_size=3,
        candidate_selection_strategy="pareto",
        use_merge=True,
        max_merge_invocations=3,
        track_stats=True,
        num_threads=1,
        seed=42,
    )

    print("Starting GEPA optimization...")
    print("  (GEPA will reflect on failures and propose improved instructions)")
    print()

    optimized = gepa.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )

    # Results
    print()
    print("=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    # Evaluate optimized
    optimized_scores = []
    for i, example in enumerate(valset):
        print(f"\n  Example {i+1}: {example.scenario[:60]}...")
        try:
            pred = optimized(scenario=example.scenario, counterpart_position=example.counterpart_position)
            result = negotiation_metric(example, pred)
            score = result['score'] if isinstance(result, dict) else result
            optimized_scores.append(score)
            print(f"    Score: {score:.3f}")
            print(f"    Response preview: {pred.response[:100]}...")
        except Exception as e:
            print(f"    Error: {e}")
            optimized_scores.append(0.0)

    optimized_avg = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0
    improvement = ((optimized_avg / baseline_avg) - 1) * 100 if baseline_avg > 0 else 0

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline average:  {baseline_avg:.3f}")
    print(f"  Optimized average: {optimized_avg:.3f}")
    print(f"  Improvement:       {improvement:+.1f}%")

    # Show evolved instruction
    print()
    print("=" * 70)
    print("EVOLVED INSTRUCTION")
    print("=" * 70)

    # Access the optimized predictor's instruction
    for name, predictor in optimized.named_predictors():
        if hasattr(predictor, 'signature'):
            print(f"\n  Predictor: {name}")
            print(f"  Instruction: {predictor.signature.__doc__}")

    # Show detailed stats if available
    if hasattr(optimized, 'detailed_results') and optimized.detailed_results:
        results = optimized.detailed_results
        print()
        print("=" * 70)
        print("GEPA STATISTICS")
        print("=" * 70)
        print(f"  Candidates explored: {len(results.candidates)}")
        print(f"  Total metric calls: {results.total_metric_calls}")
        print(f"  Best candidate index: {results.best_idx}")
        print(f"  Pareto frontier scores: {results.val_aggregate_scores[:5]}...")

    return optimized


if __name__ == "__main__":
    run_gepa_optimization()
