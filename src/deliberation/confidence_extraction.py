"""
Confidence Extraction from LLM Logprobs

Uses token-level log probabilities to extract:
- Overall response confidence
- Position-specific confidence scores
- Uncertainty indicators
- Calibrated probability estimates
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.llm import OpenRouterLM, LMResponse, TokenLogprob


# Models VERIFIED to return logprobs via OpenRouter (Jan 2026 testing)
# Many models claim support but don't actually return logprobs
LOGPROB_MODELS = [
    # OpenAI - all gpt-4o variants work
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/chatgpt-4o-latest",
    # DeepSeek - only v3.2
    "deepseek/deepseek-v3.2",
    # Qwen - only 32b
    "qwen/qwen3-32b",
    # Llama
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]


@dataclass
class ConfidenceScore:
    """Confidence score with breakdown."""
    overall: float  # 0-1, overall confidence
    mean_logprob: float
    token_confidences: List[float] = field(default_factory=list)
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    uncertainty_tokens: List[str] = field(default_factory=list)  # Tokens indicating uncertainty

    def is_uncertain(self) -> bool:
        """Check if response shows high uncertainty."""
        return self.overall < 0.5 or len(self.uncertainty_tokens) > 0


@dataclass
class PositionConfidence:
    """Confidence for a specific position/choice."""
    position_id: str
    probability: float  # Model's estimated probability for this position
    confidence: float  # How confident the model is in this estimate
    reasoning_confidence: float  # Confidence in the reasoning
    alternatives: List[Tuple[str, float]] = field(default_factory=list)


def extract_confidence(response: LMResponse) -> ConfidenceScore:
    """Extract confidence score from LLM response."""
    if response.logprobs:
        token_confs = [lp.prob for lp in response.logprobs]
        mean_lp = response.mean_logprob or 0.0

        # Find uncertainty indicators
        uncertainty_tokens = []
        uncertainty_words = ["uncertain", "maybe", "perhaps", "possibly", "unclear", "unsure", "might"]
        for lp in response.logprobs:
            if any(uw in lp.token.lower() for uw in uncertainty_words):
                uncertainty_tokens.append(lp.token)

        return ConfidenceScore(
            overall=response.get_confidence(),
            mean_logprob=mean_lp,
            token_confidences=token_confs,
            min_confidence=min(token_confs) if token_confs else 0.0,
            max_confidence=max(token_confs) if token_confs else 1.0,
            uncertainty_tokens=uncertainty_tokens,
        )

    return ConfidenceScore(
        overall=0.5,  # Default uncertainty
        mean_logprob=0.0,
    )


async def get_position_confidences(
    lm: OpenRouterLM,
    positions: List[Dict[str, str]],  # [{id, name, description}]
    topic: str,
    context: str = "",
) -> Dict[str, PositionConfidence]:
    """
    Get confidence-scored probabilities for each position.

    Uses structured prompting to elicit calibrated estimates.
    """
    position_list = "\n".join([
        f"- {p['id']}: {p['name']} - {p['description']}"
        for p in positions
    ])

    prompt = f"""Topic: {topic}
{f"Context: {context}" if context else ""}

Positions to evaluate:
{position_list}

For each position, estimate:
1. Your probability that this is the best approach (0.0 to 1.0)
2. Your confidence in this estimate (low/medium/high)

Respond in this exact format for each position:
POSITION_ID: probability confidence
Example: safety_first: 0.45 high

Your estimates should sum to approximately 1.0."""

    response = await lm(prompt, max_tokens=500)

    # Parse response
    results = {}
    overall_conf = extract_confidence(response)

    for line in response.text.strip().split("\n"):
        # Match formats: "safety_first: 0.45 high", "- safety_first: 0.45 high"
        match = re.match(r"[-*]?\s*(\w+):\s*([\d.]+)\s*(low|medium|high)?", line.strip(), re.IGNORECASE)
        if match:
            pos_id = match.group(1).lower()
            prob = float(match.group(2))
            conf_word = (match.group(3) or "medium").lower()

            conf_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
            conf = conf_map.get(conf_word, 0.6)

            # Adjust confidence by overall response confidence
            adjusted_conf = conf * overall_conf.overall

            results[pos_id] = PositionConfidence(
                position_id=pos_id,
                probability=max(0, min(1, prob)),
                confidence=adjusted_conf,
                reasoning_confidence=overall_conf.overall,
            )

    return results


async def get_conviction_with_confidence(
    lm: OpenRouterLM,
    positions: List[Dict[str, str]],
    topic: str,
    total_points: int = 100,
) -> Dict[str, Tuple[float, float]]:
    """
    Get conviction allocation with confidence scores.

    Returns: {position_id: (conviction_points, confidence)}
    """
    position_list = "\n".join([
        f"- {p['id']}: {p['name']}"
        for p in positions
    ])

    prompt = f"""Topic: {topic}

You have {total_points} conviction points to allocate across these positions.
Allocate more points to positions you believe in more strongly.

Positions:
{position_list}

For each position, provide points (summing to {total_points}):
POSITION_ID: POINTS

Be decisive - don't split evenly unless you genuinely have no preference."""

    response = await lm(prompt, max_tokens=150)
    conf = extract_confidence(response)

    results = {}
    for line in response.text.strip().split("\n"):
        # Match formats: "safety: 70", "- safety: 70", "safety: 70 points"
        match = re.match(r"[-*]?\s*(\w+):\s*(\d+)", line.strip())
        if match:
            pos_id = match.group(1).lower()
            points = int(match.group(2))
            results[pos_id] = (points, conf.overall)

    return results


async def get_binary_choice_with_logprobs(
    lm: OpenRouterLM,
    question: str,
    option_a: str,
    option_b: str,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Get binary choice with logprob-derived confidence.

    Returns: (chosen_option, confidence, {option: probability})
    """
    prompt = f"""{question}

Options:
A: {option_a}
B: {option_b}

Reply with just the letter (A or B) of your choice."""

    response = await lm(prompt, max_tokens=5, temperature=0.0)

    # Parse choice
    text = response.text.strip().upper()
    if "A" in text:
        choice = "A"
    elif "B" in text:
        choice = "B"
    else:
        choice = "A"  # Default

    # Get confidence from logprobs
    confidence = response.get_confidence()

    # Try to extract probabilities from logprobs alternatives
    probs = {"A": 0.5, "B": 0.5}
    if response.logprobs:
        first_token = response.logprobs[0]
        for alt_token, alt_logprob in first_token.top_alternatives:
            if "A" in alt_token.upper():
                probs["A"] = math.exp(alt_logprob)
            elif "B" in alt_token.upper():
                probs["B"] = math.exp(alt_logprob)

        # Normalize
        total = probs["A"] + probs["B"]
        if total > 0:
            probs = {k: v/total for k, v in probs.items()}

    return choice, confidence, probs


class ConfidenceWeightedVoting:
    """Voting mechanism weighted by logprob-derived confidence."""

    def __init__(self, positions: List[Dict[str, str]], topic: str):
        self.positions = {p["id"]: p for p in positions}
        self.topic = topic
        self.votes: Dict[str, List[Tuple[str, float, float]]] = {
            p["id"]: [] for p in positions
        }  # position_id -> [(agent_id, conviction, confidence)]

    async def collect_vote(
        self,
        agent_id: str,
        lm: OpenRouterLM,
    ):
        """Collect confidence-weighted vote from an agent."""
        convictions = await get_conviction_with_confidence(
            lm,
            list(self.positions.values()),
            self.topic,
        )

        for pos_id, (points, confidence) in convictions.items():
            if pos_id in self.votes:
                self.votes[pos_id].append((agent_id, points, confidence))

    def get_results(self) -> Dict[str, Any]:
        """Get confidence-weighted results."""
        scores = {}
        contributions = {}

        for pos_id in self.positions:
            votes = self.votes[pos_id]
            if not votes:
                scores[pos_id] = 0.0
                continue

            # Confidence-weighted sum
            weighted_sum = sum(conv * conf for _, conv, conf in votes)
            total_weight = sum(conf for _, _, conf in votes)

            scores[pos_id] = weighted_sum / total_weight if total_weight > 0 else 0.0

            # Track contributions
            for agent_id, conv, conf in votes:
                if agent_id not in contributions:
                    contributions[agent_id] = {}
                contributions[agent_id][pos_id] = {
                    "conviction": conv,
                    "confidence": conf,
                    "weighted_contribution": conv * conf,
                }

        # Normalize scores
        total = sum(scores.values()) or 1
        scores = {k: v/total for k, v in scores.items()}

        winner_id = max(scores, key=scores.get)

        return {
            "winner": winner_id,
            "scores": scores,
            "contributions": contributions,
            "mechanism": "confidence_weighted_voting",
        }


# Convenience function for quick confidence-weighted deliberation
async def quick_deliberation(
    topic: str,
    positions: List[Dict[str, str]],
    num_samples: int = 3,
    model: str = "openai/gpt-4o-mini",  # Best for logprobs via OpenRouter
) -> Dict[str, Any]:
    """
    Run quick confidence-weighted deliberation.

    Samples multiple responses and aggregates with confidence weighting.
    """
    lm = OpenRouterLM(model=model, enable_logprobs=True)

    all_convictions = []

    for i in range(num_samples):
        convictions = await get_conviction_with_confidence(
            lm, positions, topic
        )
        all_convictions.append(convictions)

    # Aggregate
    aggregated = {p["id"]: [] for p in positions}
    for conv_dict in all_convictions:
        for pos_id, (points, conf) in conv_dict.items():
            if pos_id in aggregated:
                aggregated[pos_id].append((points, conf))

    # Confidence-weighted average
    final_scores = {}
    for pos_id, samples in aggregated.items():
        if samples:
            weighted_sum = sum(p * c for p, c in samples)
            weight_sum = sum(c for _, c in samples)
            final_scores[pos_id] = weighted_sum / weight_sum if weight_sum > 0 else 0
        else:
            final_scores[pos_id] = 0

    # Normalize
    total = sum(final_scores.values()) or 1
    final_scores = {k: v/total for k, v in final_scores.items()}

    winner = max(final_scores, key=final_scores.get)

    return {
        "winner": winner,
        "scores": final_scores,
        "num_samples": num_samples,
        "raw_samples": all_convictions,
    }
