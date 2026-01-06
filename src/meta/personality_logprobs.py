"""Logprob-based personality analysis and steering.

Uses token-level log probabilities to:
- Probe model personality alignment
- Measure trait expression strength
- Detect personality drift
- Score response authenticity
- Steer generation toward personality
"""

from __future__ import annotations

import math
import hashlib
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .personality import (
    Personality,
    TraitDimension,
    TraitProfile,
    CognitiveStyle,
    ValueOrientation,
    VoicePattern,
    ToneRegister,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Logprob Data Structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TokenLogprob:
    """Single token with its logprob and alternatives."""
    token: str
    logprob: float
    top_logprobs: Dict[str, float] = field(default_factory=dict)  # alternative tokens
    position: int = 0

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)

    @property
    def entropy(self) -> float:
        """Shannon entropy over top alternatives."""
        if not self.top_logprobs:
            return 0.0
        probs = [math.exp(lp) for lp in self.top_logprobs.values()]
        total = sum(probs)
        if total == 0:
            return 0.0
        probs = [p / total for p in probs]
        return -sum(p * math.log(p + 1e-10) for p in probs)


@dataclass
class ResponseLogprobs:
    """Logprobs for an entire response."""
    tokens: List[TokenLogprob] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    personality_name: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mean_logprob(self) -> float:
        if not self.tokens:
            return 0.0
        return statistics.mean(t.logprob for t in self.tokens)

    @property
    def mean_entropy(self) -> float:
        if not self.tokens:
            return 0.0
        return statistics.mean(t.entropy for t in self.tokens)

    @property
    def perplexity(self) -> float:
        """Perplexity of the response."""
        if not self.tokens:
            return 0.0
        avg_logprob = self.mean_logprob
        return math.exp(-avg_logprob)

    def get_low_confidence_spans(self, threshold: float = -3.0) -> List[Tuple[int, int, str]]:
        """Find spans where model was uncertain."""
        spans = []
        in_span = False
        start = 0
        span_tokens = []

        for i, tok in enumerate(self.tokens):
            if tok.logprob < threshold:
                if not in_span:
                    in_span = True
                    start = i
                    span_tokens = []
                span_tokens.append(tok.token)
            else:
                if in_span:
                    spans.append((start, i, "".join(span_tokens)))
                    in_span = False

        if in_span:
            spans.append((start, len(self.tokens), "".join(span_tokens)))

        return spans


# ─────────────────────────────────────────────────────────────────────────────
# Trait Lexicons - Words indicative of personality traits
# ─────────────────────────────────────────────────────────────────────────────


class TraitLexicon:
    """Lexicons of words/phrases indicative of personality traits."""

    # High trait indicators (presence suggests high trait)
    HIGH_INDICATORS: Dict[TraitDimension, Set[str]] = {
        TraitDimension.ANALYTICITY: {
            "analyze", "therefore", "consequently", "logically", "systematically",
            "evidence", "data", "framework", "structure", "methodology",
            "hypothesis", "correlate", "derive", "deduce", "infer",
            "quantify", "measure", "evaluate", "assess", "criterion",
        },
        TraitDimension.CREATIVITY: {
            "imagine", "innovative", "novel", "creative", "unique",
            "possibility", "explore", "experiment", "unconventional", "original",
            "inspire", "vision", "reimagine", "transform", "invent",
            "synthesize", "blend", "fusion", "emergent", "breakthrough",
        },
        TraitDimension.ASSERTIVENESS: {
            "certainly", "definitely", "absolutely", "clearly", "obviously",
            "must", "should", "need", "require", "essential",
            "decisive", "direct", "straightforward", "unambiguous", "firm",
            "confident", "assured", "convinced", "determined", "resolute",
        },
        TraitDimension.AGREEABLENESS: {
            "understand", "appreciate", "respect", "consider", "acknowledge",
            "together", "collaborate", "share", "support", "help",
            "empathize", "compassion", "kind", "gentle", "patient",
            "inclusive", "harmonious", "cooperative", "mutual", "consensus",
        },
        TraitDimension.SKEPTICISM: {
            "however", "although", "but", "yet", "nevertheless",
            "question", "doubt", "uncertain", "unclear", "debatable",
            "caveat", "limitation", "assumption", "allegedly", "supposedly",
            "scrutinize", "verify", "validate", "challenge", "critique",
        },
        TraitDimension.OPENNESS: {
            "interesting", "fascinating", "curious", "wonder", "explore",
            "diverse", "variety", "perspective", "alternative", "possibility",
            "abstract", "theoretical", "philosophical", "conceptual", "nuanced",
            "complex", "multifaceted", "layered", "rich", "depth",
        },
        TraitDimension.CONSCIENTIOUSNESS: {
            "careful", "thorough", "detailed", "precise", "accurate",
            "organized", "systematic", "methodical", "rigorous", "diligent",
            "complete", "comprehensive", "exhaustive", "meticulous", "exact",
            "reliable", "consistent", "dependable", "responsible", "disciplined",
        },
        TraitDimension.HUMOR: {
            "haha", "funny", "amusing", "witty", "clever",
            "joke", "playful", "lighthearted", "tongue-in-cheek", "ironic",
            "entertaining", "whimsical", "comical", "humorous", "jest",
        },
        TraitDimension.FORMALITY: {
            "furthermore", "moreover", "regarding", "concerning", "pursuant",
            "accordingly", "hereby", "thereof", "whereas", "notwithstanding",
            "respectfully", "formally", "professionally", "appropriately", "properly",
        },
        TraitDimension.PATIENCE: {
            "gradually", "slowly", "step-by-step", "eventually", "over time",
            "patience", "careful", "deliberate", "measured", "unhurried",
            "thoroughly", "comprehensively", "extensively", "in-depth", "detailed",
        },
    }

    # Low trait indicators (presence suggests low trait)
    LOW_INDICATORS: Dict[TraitDimension, Set[str]] = {
        TraitDimension.ANALYTICITY: {
            "feel", "sense", "intuition", "gut", "hunch",
            "somehow", "maybe", "perhaps", "probably", "might",
        },
        TraitDimension.CREATIVITY: {
            "standard", "conventional", "traditional", "typical", "normal",
            "usual", "common", "ordinary", "regular", "routine",
        },
        TraitDimension.ASSERTIVENESS: {
            "maybe", "perhaps", "possibly", "might", "could",
            "uncertain", "unsure", "hesitant", "tentative", "cautious",
        },
        TraitDimension.AGREEABLENESS: {
            "disagree", "reject", "refuse", "deny", "oppose",
            "incorrect", "wrong", "mistaken", "flawed", "erroneous",
        },
        TraitDimension.SKEPTICISM: {
            "certainly", "definitely", "obviously", "clearly", "undoubtedly",
            "trust", "believe", "accept", "assume", "presume",
        },
        TraitDimension.FORMALITY: {
            "gonna", "wanna", "kinda", "sorta", "yeah",
            "ok", "cool", "awesome", "stuff", "things",
            "hey", "hi", "sup", "yo", "dude",
        },
    }

    # Phrase patterns (multi-word indicators)
    PHRASE_PATTERNS: Dict[TraitDimension, List[str]] = {
        TraitDimension.ANALYTICITY: [
            "based on", "according to", "in terms of", "with respect to",
            "it follows that", "this suggests", "the data shows", "evidence indicates",
        ],
        TraitDimension.AGREEABLENESS: [
            "I understand", "that makes sense", "good point", "I appreciate",
            "thank you for", "I see what you mean", "that's a valid",
        ],
        TraitDimension.SKEPTICISM: [
            "I'm not sure", "it's worth noting", "one could argue",
            "on the other hand", "that said", "to be fair",
        ],
        TraitDimension.CREATIVITY: [
            "what if", "imagine if", "another way to", "we could also",
            "thinking outside", "a fresh perspective", "novel approach",
        ],
    }

    @classmethod
    def get_trait_tokens(cls, trait: TraitDimension, high: bool = True) -> Set[str]:
        """Get tokens indicative of high or low trait."""
        if high:
            return cls.HIGH_INDICATORS.get(trait, set())
        return cls.LOW_INDICATORS.get(trait, set())


# ─────────────────────────────────────────────────────────────────────────────
# Personality Probing
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TraitProbeResult:
    """Result of probing a specific trait."""
    trait: TraitDimension
    measured_value: float  # 0-1, inferred from logprobs
    confidence: float  # How confident we are in measurement
    high_indicator_hits: int
    low_indicator_hits: int
    supporting_tokens: List[Tuple[str, float]]  # (token, logprob)
    contradicting_tokens: List[Tuple[str, float]]


@dataclass
class PersonalityProbeResult:
    """Complete personality probe from logprobs."""
    trait_results: Dict[TraitDimension, TraitProbeResult] = field(default_factory=dict)
    overall_alignment: float = 0.0  # How well response matches expected personality
    authenticity_score: float = 0.0  # How natural the personality expression is
    drift_indicators: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PersonalityProber:
    """Probe personality from response logprobs."""

    def __init__(self, personality: Optional[Personality] = None):
        self.personality = personality
        self.history: List[PersonalityProbeResult] = []

    def probe(
        self,
        response_logprobs: ResponseLogprobs,
        personality: Optional[Personality] = None,
    ) -> PersonalityProbeResult:
        """Probe personality traits from response logprobs."""
        personality = personality or self.personality

        result = PersonalityProbeResult()

        # Extract all tokens for analysis
        tokens_lower = [t.token.lower().strip() for t in response_logprobs.tokens]
        token_logprobs = {t.token.lower().strip(): t.logprob for t in response_logprobs.tokens}

        # Probe each trait
        for trait in TraitDimension:
            trait_result = self._probe_trait(trait, tokens_lower, token_logprobs, response_logprobs)
            result.trait_results[trait] = trait_result

        # Calculate overall alignment if we have a target personality
        if personality:
            result.overall_alignment = self._calculate_alignment(result, personality)
            result.drift_indicators = self._detect_drift(result, personality)

        # Calculate authenticity (natural expression vs forced)
        result.authenticity_score = self._calculate_authenticity(response_logprobs, result)

        self.history.append(result)
        return result

    def _probe_trait(
        self,
        trait: TraitDimension,
        tokens: List[str],
        token_logprobs: Dict[str, float],
        response_logprobs: ResponseLogprobs,
    ) -> TraitProbeResult:
        """Probe a single trait from tokens."""
        high_indicators = TraitLexicon.get_trait_tokens(trait, high=True)
        low_indicators = TraitLexicon.get_trait_tokens(trait, high=False)

        high_hits = []
        low_hits = []

        for token in tokens:
            # Check direct matches
            if token in high_indicators:
                lp = token_logprobs.get(token, -5.0)
                high_hits.append((token, lp))
            if token in low_indicators:
                lp = token_logprobs.get(token, -5.0)
                low_hits.append((token, lp))

            # Check partial matches (token contains indicator)
            for ind in high_indicators:
                if ind in token and token not in [h[0] for h in high_hits]:
                    lp = token_logprobs.get(token, -5.0)
                    high_hits.append((token, lp))
                    break

        # Calculate measured value
        high_weight = sum(math.exp(lp) for _, lp in high_hits) if high_hits else 0
        low_weight = sum(math.exp(lp) for _, lp in low_hits) if low_hits else 0
        total = high_weight + low_weight + 1e-10

        measured = 0.5 + 0.5 * (high_weight - low_weight) / (total + 1e-10)
        measured = max(0.0, min(1.0, measured))

        # Confidence based on number of indicators found
        total_hits = len(high_hits) + len(low_hits)
        confidence = min(1.0, total_hits / 10.0)  # Max confidence at 10+ hits

        return TraitProbeResult(
            trait=trait,
            measured_value=measured,
            confidence=confidence,
            high_indicator_hits=len(high_hits),
            low_indicator_hits=len(low_hits),
            supporting_tokens=high_hits[:5],
            contradicting_tokens=low_hits[:5],
        )

    def _calculate_alignment(
        self,
        probe_result: PersonalityProbeResult,
        personality: Personality,
    ) -> float:
        """Calculate how well probed traits align with target personality."""
        alignments = []

        for trait, result in probe_result.trait_results.items():
            if result.confidence < 0.1:
                continue  # Skip low-confidence measurements

            target = personality.traits.get_trait(trait)
            measured = result.measured_value

            # Weighted alignment (higher confidence = more weight)
            diff = abs(target - measured)
            alignment = 1.0 - diff
            alignments.append(alignment * result.confidence)

        if not alignments:
            return 0.5

        return sum(alignments) / len(alignments)

    def _detect_drift(
        self,
        probe_result: PersonalityProbeResult,
        personality: Personality,
    ) -> List[str]:
        """Detect traits that are drifting from target."""
        drift = []

        for trait, result in probe_result.trait_results.items():
            if result.confidence < 0.2:
                continue

            target = personality.traits.get_trait(trait)
            measured = result.measured_value

            # Significant drift threshold
            if abs(target - measured) > 0.3:
                direction = "higher" if measured > target else "lower"
                drift.append(f"{trait.value}: expressing {direction} than target")

        return drift

    def _calculate_authenticity(
        self,
        response_logprobs: ResponseLogprobs,
        probe_result: PersonalityProbeResult,
    ) -> float:
        """Calculate how natural/authentic the personality expression is.

        Low authenticity = model is being forced into unnatural patterns
        High authenticity = personality flows naturally from model
        """
        # High entropy on personality-indicative tokens suggests struggle
        trait_tokens = set()
        for trait in TraitDimension:
            trait_tokens.update(TraitLexicon.get_trait_tokens(trait, True))
            trait_tokens.update(TraitLexicon.get_trait_tokens(trait, False))

        trait_entropies = []
        other_entropies = []

        for tok in response_logprobs.tokens:
            if tok.token.lower().strip() in trait_tokens:
                trait_entropies.append(tok.entropy)
            else:
                other_entropies.append(tok.entropy)

        if not trait_entropies:
            return 0.5

        # If trait tokens have much higher entropy than others,
        # model is uncertain about personality expression
        trait_mean = statistics.mean(trait_entropies)
        other_mean = statistics.mean(other_entropies) if other_entropies else trait_mean

        # Lower relative entropy on trait tokens = more authentic
        if other_mean > 0:
            ratio = trait_mean / (other_mean + 1e-10)
            authenticity = 1.0 / (1.0 + ratio)  # Higher when ratio is low
        else:
            authenticity = 0.5

        return authenticity


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Personality Analysis
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ContrastiveAnalysis:
    """Analysis of logprob differences between personalities."""
    personality_a: str
    personality_b: str
    divergent_tokens: List[Tuple[str, float, float]]  # (token, logprob_a, logprob_b)
    mean_divergence: float
    max_divergence: float
    personality_signature_tokens: Dict[str, List[str]]  # personality -> distinctive tokens


class ContrastiveAnalyzer:
    """Compare logprobs across different personalities."""

    def __init__(self):
        self.cache: Dict[str, ResponseLogprobs] = {}

    def compare(
        self,
        logprobs_a: ResponseLogprobs,
        logprobs_b: ResponseLogprobs,
        personality_a: str = "A",
        personality_b: str = "B",
    ) -> ContrastiveAnalysis:
        """Compare logprobs from two personalities on same content."""

        # Align tokens (they should be similar if same prompt)
        tokens_a = {t.token: t.logprob for t in logprobs_a.tokens}
        tokens_b = {t.token: t.logprob for t in logprobs_b.tokens}

        common_tokens = set(tokens_a.keys()) & set(tokens_b.keys())

        divergent = []
        signature_a = []
        signature_b = []

        for token in common_tokens:
            lp_a = tokens_a[token]
            lp_b = tokens_b[token]
            diff = lp_a - lp_b

            if abs(diff) > 0.5:  # Significant divergence
                divergent.append((token, lp_a, lp_b))

                if diff > 0.5:  # A prefers this token
                    signature_a.append(token)
                else:  # B prefers this token
                    signature_b.append(token)

        divergent.sort(key=lambda x: abs(x[1] - x[2]), reverse=True)

        divergences = [abs(d[1] - d[2]) for d in divergent]

        return ContrastiveAnalysis(
            personality_a=personality_a,
            personality_b=personality_b,
            divergent_tokens=divergent[:20],  # Top 20
            mean_divergence=statistics.mean(divergences) if divergences else 0.0,
            max_divergence=max(divergences) if divergences else 0.0,
            personality_signature_tokens={
                personality_a: signature_a[:10],
                personality_b: signature_b[:10],
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Personality Steering with Logprobs
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SteeringBias:
    """Token-level bias for steering generation toward personality."""
    token: str
    bias: float  # Positive = increase probability, negative = decrease
    trait: Optional[TraitDimension] = None
    reason: str = ""


class PersonalitySteering:
    """Generate logit biases to steer generation toward personality."""

    def __init__(self, personality: Personality):
        self.personality = personality
        self._bias_cache: Dict[str, float] = {}

    def generate_biases(
        self,
        strength: float = 1.0,
        max_biases: int = 100,
    ) -> Dict[str, float]:
        """Generate token biases based on personality traits.

        Args:
            strength: How strong the bias should be (0-2)
            max_biases: Maximum number of token biases to return

        Returns:
            Dict mapping tokens to bias values
        """
        biases: List[SteeringBias] = []

        for trait in TraitDimension:
            trait_value = self.personality.traits.get_trait(trait)

            # Get indicator tokens
            high_tokens = TraitLexicon.get_trait_tokens(trait, high=True)
            low_tokens = TraitLexicon.get_trait_tokens(trait, high=False)

            # High trait = boost high indicators, reduce low indicators
            # Low trait = opposite
            high_bias = (trait_value - 0.5) * 2 * strength  # -1 to 1 scaled by strength
            low_bias = -high_bias

            for token in high_tokens:
                if abs(high_bias) > 0.1:
                    biases.append(SteeringBias(
                        token=token,
                        bias=high_bias,
                        trait=trait,
                        reason=f"High {trait.value} indicator",
                    ))

            for token in low_tokens:
                if abs(low_bias) > 0.1:
                    biases.append(SteeringBias(
                        token=token,
                        bias=low_bias,
                        trait=trait,
                        reason=f"Low {trait.value} indicator",
                    ))

        # Add voice pattern biases
        voice_biases = self._generate_voice_biases(strength)
        biases.extend(voice_biases)

        # Aggregate biases for same token
        token_biases: Dict[str, float] = defaultdict(float)
        for b in biases:
            token_biases[b.token] += b.bias

        # Clip and limit
        result = {}
        for token, bias in sorted(token_biases.items(), key=lambda x: abs(x[1]), reverse=True):
            if len(result) >= max_biases:
                break
            clipped = max(-5.0, min(5.0, bias))
            if abs(clipped) > 0.1:
                result[token] = clipped

        return result

    def _generate_voice_biases(self, strength: float) -> List[SteeringBias]:
        """Generate biases from voice patterns."""
        biases = []
        voice = self.personality.voice

        # Formality biases
        if voice.primary_tone == ToneRegister.FORMAL:
            formal_tokens = {"furthermore", "moreover", "regarding", "accordingly"}
            for t in formal_tokens:
                biases.append(SteeringBias(token=t, bias=0.5 * strength, reason="Formal tone"))
        elif voice.primary_tone in [ToneRegister.CASUAL, ToneRegister.PLAYFUL]:
            casual_tokens = {"yeah", "cool", "awesome", "hey"}
            for t in casual_tokens:
                biases.append(SteeringBias(token=t, bias=0.5 * strength, reason="Casual tone"))

        # Hedging biases
        hedging_tokens = {"perhaps", "maybe", "possibly", "might", "could"}
        hedging_bias = (voice.use_hedging - 0.5) * 2 * strength
        for t in hedging_tokens:
            biases.append(SteeringBias(token=t, bias=hedging_bias, reason="Hedging level"))

        # Emphasis biases
        emphasis_tokens = {"certainly", "definitely", "absolutely", "clearly"}
        emphasis_bias = (voice.use_emphasis - 0.5) * 2 * strength
        for t in emphasis_tokens:
            biases.append(SteeringBias(token=t, bias=emphasis_bias, reason="Emphasis level"))

        return biases

    def adjust_for_context(
        self,
        base_biases: Dict[str, float],
        context: str,
        context_type: str = "general",
    ) -> Dict[str, float]:
        """Adjust biases based on conversation context."""
        adjusted = base_biases.copy()

        # Technical context: boost analytical tokens
        if context_type == "technical":
            tech_boost = {"analyze", "implement", "function", "method", "algorithm"}
            for t in tech_boost:
                adjusted[t] = adjusted.get(t, 0) + 0.3

        # Creative context: boost creative tokens
        elif context_type == "creative":
            creative_boost = {"imagine", "create", "design", "vision", "inspire"}
            for t in creative_boost:
                adjusted[t] = adjusted.get(t, 0) + 0.3

        # Empathetic context: boost agreeable tokens
        elif context_type == "empathetic":
            empathy_boost = {"understand", "feel", "appreciate", "support", "help"}
            for t in empathy_boost:
                adjusted[t] = adjusted.get(t, 0) + 0.3

        return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# Personality Drift Tracker
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DriftMeasurement:
    """Single drift measurement over time."""
    timestamp: datetime
    trait_values: Dict[TraitDimension, float]
    overall_drift: float
    authenticity: float


class PersonalityDriftTracker:
    """Track personality consistency over time using logprobs."""

    def __init__(self, target_personality: Personality, window_size: int = 50):
        self.target = target_personality
        self.window_size = window_size
        self.measurements: List[DriftMeasurement] = []
        self.prober = PersonalityProber(target_personality)

    def track(self, response_logprobs: ResponseLogprobs) -> DriftMeasurement:
        """Track drift from a new response."""
        probe = self.prober.probe(response_logprobs)

        trait_values = {
            trait: result.measured_value
            for trait, result in probe.trait_results.items()
        }

        # Calculate drift from target
        drifts = []
        for trait, measured in trait_values.items():
            target = self.target.traits.get_trait(trait)
            drifts.append(abs(measured - target))

        measurement = DriftMeasurement(
            timestamp=datetime.utcnow(),
            trait_values=trait_values,
            overall_drift=statistics.mean(drifts) if drifts else 0.0,
            authenticity=probe.authenticity_score,
        )

        self.measurements.append(measurement)

        # Keep window bounded
        if len(self.measurements) > self.window_size:
            self.measurements = self.measurements[-self.window_size:]

        return measurement

    def get_drift_trend(self) -> Dict[str, float]:
        """Get drift trends over recent measurements."""
        if len(self.measurements) < 2:
            return {"trend": 0.0, "volatility": 0.0}

        recent = self.measurements[-10:]
        drifts = [m.overall_drift for m in recent]

        # Simple linear trend
        if len(drifts) >= 2:
            trend = (drifts[-1] - drifts[0]) / len(drifts)
        else:
            trend = 0.0

        return {
            "trend": trend,  # Positive = increasing drift
            "volatility": statistics.stdev(drifts) if len(drifts) > 1 else 0.0,
            "mean_drift": statistics.mean(drifts),
            "mean_authenticity": statistics.mean(m.authenticity for m in recent),
        }

    def get_problem_traits(self, threshold: float = 0.25) -> List[TraitDimension]:
        """Identify traits with consistent drift."""
        if not self.measurements:
            return []

        recent = self.measurements[-10:]
        problem_traits = []

        for trait in TraitDimension:
            target = self.target.traits.get_trait(trait)
            measured_values = [
                m.trait_values.get(trait, 0.5) for m in recent
                if trait in m.trait_values
            ]

            if not measured_values:
                continue

            mean_measured = statistics.mean(measured_values)
            if abs(mean_measured - target) > threshold:
                problem_traits.append(trait)

        return problem_traits


# ─────────────────────────────────────────────────────────────────────────────
# Integration Helper
# ─────────────────────────────────────────────────────────────────────────────


class LogprobPersonalityAnalyzer:
    """High-level interface for logprob-based personality analysis."""

    def __init__(self, personality: Personality):
        self.personality = personality
        self.prober = PersonalityProber(personality)
        self.steering = PersonalitySteering(personality)
        self.drift_tracker = PersonalityDriftTracker(personality)
        self.contrastive = ContrastiveAnalyzer()

    def analyze_response(
        self,
        response_logprobs: ResponseLogprobs,
    ) -> Dict[str, Any]:
        """Complete analysis of a response."""
        # Probe personality
        probe = self.prober.probe(response_logprobs)

        # Track drift
        drift = self.drift_tracker.track(response_logprobs)

        # Get drift trends
        trends = self.drift_tracker.get_drift_trend()

        # Identify problems
        problem_traits = self.drift_tracker.get_problem_traits()

        return {
            "alignment": probe.overall_alignment,
            "authenticity": probe.authenticity_score,
            "drift": drift.overall_drift,
            "drift_trend": trends,
            "problem_traits": [t.value for t in problem_traits],
            "trait_measurements": {
                t.value: {
                    "measured": r.measured_value,
                    "target": self.personality.traits.get_trait(t),
                    "confidence": r.confidence,
                }
                for t, r in probe.trait_results.items()
            },
            "low_confidence_spans": response_logprobs.get_low_confidence_spans(),
            "perplexity": response_logprobs.perplexity,
        }

    def get_steering_biases(
        self,
        strength: float = 1.0,
        context_type: str = "general",
    ) -> Dict[str, float]:
        """Get token biases for steering generation."""
        base_biases = self.steering.generate_biases(strength)
        return self.steering.adjust_for_context(base_biases, "", context_type)

    def should_recalibrate(self) -> Tuple[bool, str]:
        """Check if personality needs recalibration."""
        trends = self.drift_tracker.get_drift_trend()

        if trends.get("mean_drift", 0) > 0.3:
            return True, "High mean drift from target personality"

        if trends.get("trend", 0) > 0.05:
            return True, "Drift is increasing over time"

        if trends.get("mean_authenticity", 1) < 0.4:
            return True, "Low authenticity - personality feels forced"

        problem_traits = self.drift_tracker.get_problem_traits()
        if len(problem_traits) >= 3:
            return True, f"Multiple problematic traits: {[t.value for t in problem_traits]}"

        return False, "Personality expression is stable"


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def parse_openai_logprobs(logprobs_data: dict) -> ResponseLogprobs:
    """Parse OpenAI-format logprobs into our structure."""
    tokens = []

    content = logprobs_data.get("content", [])
    for i, item in enumerate(content):
        top_logprobs = {}
        for alt in item.get("top_logprobs", []):
            top_logprobs[alt["token"]] = alt["logprob"]

        tokens.append(TokenLogprob(
            token=item.get("token", ""),
            logprob=item.get("logprob", 0.0),
            top_logprobs=top_logprobs,
            position=i,
        ))

    return ResponseLogprobs(tokens=tokens)


def parse_ollama_logprobs(response: dict) -> ResponseLogprobs:
    """Parse Ollama-format response with logprobs."""
    tokens = []

    # Ollama returns logprobs differently
    if "logprobs" in response:
        for i, (token, logprob) in enumerate(zip(
            response.get("tokens", []),
            response.get("logprobs", [])
        )):
            tokens.append(TokenLogprob(
                token=token,
                logprob=logprob,
                position=i,
            ))

    return ResponseLogprobs(
        tokens=tokens,
        model=response.get("model", ""),
    )


def create_analyzer(personality: Personality) -> LogprobPersonalityAnalyzer:
    """Create a logprob analyzer for a personality."""
    return LogprobPersonalityAnalyzer(personality)
