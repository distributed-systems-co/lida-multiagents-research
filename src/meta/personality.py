"""Personality crystallization system for distinctive agent identities.

Provides techniques for developing and reinforcing unique personalities:
- Trait dimensions (Big Five, values, cognitive styles)
- Voice patterns (communication style, vocabulary, tone)
- Behavioral tendencies (decision heuristics, interaction patterns)
- Crystallization through experience and feedback
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trait Dimensions
# ─────────────────────────────────────────────────────────────────────────────


class TraitDimension(str, Enum):
    """Big Five personality dimensions plus extensions."""
    # Big Five
    OPENNESS = "openness"                    # Curiosity, creativity, novelty-seeking
    CONSCIENTIOUSNESS = "conscientiousness"  # Organization, discipline, reliability
    EXTRAVERSION = "extraversion"            # Sociability, assertiveness, energy
    AGREEABLENESS = "agreeableness"          # Cooperation, trust, empathy
    NEUROTICISM = "neuroticism"              # Emotional volatility, anxiety, sensitivity

    # Extended dimensions
    ANALYTICITY = "analyticity"              # Logical vs intuitive reasoning
    ASSERTIVENESS = "assertiveness"          # Directness in communication
    CREATIVITY = "creativity"                # Novel idea generation
    SKEPTICISM = "skepticism"                # Critical evaluation tendency
    PATIENCE = "patience"                    # Tolerance for delay/complexity
    HUMOR = "humor"                          # Wit and playfulness
    FORMALITY = "formality"                  # Casual vs formal communication


class CognitiveStyle(str, Enum):
    """Cognitive processing preferences."""
    ANALYTICAL = "analytical"       # Step-by-step logical analysis
    INTUITIVE = "intuitive"         # Pattern recognition, gut feelings
    SYSTEMATIC = "systematic"       # Structured, methodical approach
    EXPLORATORY = "exploratory"     # Open-ended investigation
    CONVERGENT = "convergent"       # Narrowing to single solution
    DIVERGENT = "divergent"         # Generating multiple alternatives
    CONCRETE = "concrete"           # Focus on specifics, examples
    ABSTRACT = "abstract"           # Focus on principles, theories


class ValueOrientation(str, Enum):
    """Core value orientations."""
    TRUTH = "truth"                 # Accuracy, honesty, correctness
    EFFICIENCY = "efficiency"       # Speed, optimization, pragmatism
    THOROUGHNESS = "thoroughness"   # Completeness, depth, rigor
    CLARITY = "clarity"             # Understandability, simplicity
    NOVELTY = "novelty"             # Innovation, originality
    TRADITION = "tradition"         # Established methods, stability
    AUTONOMY = "autonomy"           # Independence, self-direction
    COLLABORATION = "collaboration" # Teamwork, shared effort
    EXCELLENCE = "excellence"       # High standards, quality
    ACCESSIBILITY = "accessibility" # Inclusivity, ease of understanding


@dataclass
class TraitProfile:
    """A personality trait profile with dimension scores."""

    dimensions: Dict[TraitDimension, float] = field(default_factory=dict)
    cognitive_styles: Dict[CognitiveStyle, float] = field(default_factory=dict)
    values: Dict[ValueOrientation, float] = field(default_factory=dict)

    # Crystallization tracking
    stability: float = 0.0  # 0-1, how crystallized/stable the personality is
    interactions: int = 0
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        # Initialize with neutral values if empty
        if not self.dimensions:
            for dim in TraitDimension:
                self.dimensions[dim] = 0.5
        if not self.cognitive_styles:
            for style in CognitiveStyle:
                self.cognitive_styles[style] = 0.5
        if not self.values:
            for val in ValueOrientation:
                self.values[val] = 0.5

    def get_trait(self, dimension: TraitDimension) -> float:
        return self.dimensions.get(dimension, 0.5)

    def set_trait(self, dimension: TraitDimension, value: float):
        self.dimensions[dimension] = max(0.0, min(1.0, value))
        self.last_updated = datetime.utcnow()

    def adjust_trait(self, dimension: TraitDimension, delta: float, learning_rate: float = 0.1):
        """Adjust trait with learning rate and stability resistance."""
        current = self.get_trait(dimension)
        # Higher stability means more resistance to change
        effective_delta = delta * learning_rate * (1 - self.stability * 0.8)
        self.set_trait(dimension, current + effective_delta)

    def dominant_traits(self, threshold: float = 0.65, count: int = 3) -> List[Tuple[TraitDimension, float]]:
        """Get the most prominent personality traits."""
        strong = [(d, v) for d, v in self.dimensions.items() if v >= threshold or v <= 1 - threshold]
        strong.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        return strong[:count]

    def dominant_values(self, count: int = 3) -> List[Tuple[ValueOrientation, float]]:
        """Get strongest value orientations."""
        sorted_values = sorted(self.values.items(), key=lambda x: x[1], reverse=True)
        return sorted_values[:count]

    def primary_cognitive_style(self) -> CognitiveStyle:
        """Get the dominant cognitive style."""
        return max(self.cognitive_styles, key=self.cognitive_styles.get)

    def to_dict(self) -> dict:
        return {
            "dimensions": {k.value: v for k, v in self.dimensions.items()},
            "cognitive_styles": {k.value: v for k, v in self.cognitive_styles.items()},
            "values": {k.value: v for k, v in self.values.items()},
            "stability": self.stability,
            "interactions": self.interactions,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TraitProfile":
        return cls(
            dimensions={TraitDimension(k): v for k, v in data.get("dimensions", {}).items()},
            cognitive_styles={CognitiveStyle(k): v for k, v in data.get("cognitive_styles", {}).items()},
            values={ValueOrientation(k): v for k, v in data.get("values", {}).items()},
            stability=data.get("stability", 0.0),
            interactions=data.get("interactions", 0),
            last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Voice Patterns
# ─────────────────────────────────────────────────────────────────────────────


class ToneRegister(str, Enum):
    """Communication tone registers."""
    FORMAL = "formal"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    PLAYFUL = "playful"
    AUTHORITATIVE = "authoritative"
    EMPATHETIC = "empathetic"
    NEUTRAL = "neutral"
    # Extended tones
    ENTHUSIASTIC = "enthusiastic"
    THOUGHTFUL = "thoughtful"
    PROVOCATIVE = "provocative"
    DRAMATIC = "dramatic"
    INTENSE = "intense"
    POETIC = "poetic"
    MELANCHOLIC = "melancholic"
    SARCASTIC = "sarcastic"
    ENERGETIC = "energetic"


class ResponseLength(str, Enum):
    """Preferred response length."""
    TERSE = "terse"           # Minimal, direct
    CONCISE = "concise"       # Brief but complete
    MODERATE = "moderate"     # Balanced
    DETAILED = "detailed"     # Thorough explanation
    COMPREHENSIVE = "comprehensive"  # Exhaustive coverage


@dataclass
class VoicePattern:
    """Defines how an agent communicates."""

    # Tone and style
    primary_tone: ToneRegister = ToneRegister.PROFESSIONAL
    secondary_tone: Optional[ToneRegister] = None
    response_length: ResponseLength = ResponseLength.MODERATE

    # Language patterns
    vocabulary_level: float = 0.5          # 0=simple, 1=technical/academic
    sentence_complexity: float = 0.5       # 0=simple, 1=complex
    use_analogies: float = 0.5             # Tendency to use metaphors/analogies
    use_examples: float = 0.5              # Tendency to use examples
    use_hedging: float = 0.3               # Uncertainty markers ("perhaps", "might")
    use_emphasis: float = 0.5              # Strong assertions, emphasis

    # Structural patterns
    prefer_lists: float = 0.5              # Tendency to use lists/bullets
    prefer_headers: float = 0.3            # Use of section headers
    paragraph_length: float = 0.5          # 0=short, 1=long paragraphs

    # Personality markers
    signature_phrases: List[str] = field(default_factory=list)
    avoided_words: Set[str] = field(default_factory=set)
    preferred_transitions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary_tone": self.primary_tone.value,
            "secondary_tone": self.secondary_tone.value if self.secondary_tone else None,
            "response_length": self.response_length.value,
            "vocabulary_level": self.vocabulary_level,
            "sentence_complexity": self.sentence_complexity,
            "use_analogies": self.use_analogies,
            "use_examples": self.use_examples,
            "use_hedging": self.use_hedging,
            "use_emphasis": self.use_emphasis,
            "prefer_lists": self.prefer_lists,
            "prefer_headers": self.prefer_headers,
            "paragraph_length": self.paragraph_length,
            "signature_phrases": self.signature_phrases,
            "avoided_words": list(self.avoided_words),
            "preferred_transitions": self.preferred_transitions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VoicePattern":
        return cls(
            primary_tone=ToneRegister(data.get("primary_tone", "professional")),
            secondary_tone=ToneRegister(data["secondary_tone"]) if data.get("secondary_tone") else None,
            response_length=ResponseLength(data.get("response_length", "moderate")),
            vocabulary_level=data.get("vocabulary_level", 0.5),
            sentence_complexity=data.get("sentence_complexity", 0.5),
            use_analogies=data.get("use_analogies", 0.5),
            use_examples=data.get("use_examples", 0.5),
            use_hedging=data.get("use_hedging", 0.3),
            use_emphasis=data.get("use_emphasis", 0.5),
            prefer_lists=data.get("prefer_lists", 0.5),
            prefer_headers=data.get("prefer_headers", 0.3),
            paragraph_length=data.get("paragraph_length", 0.5),
            signature_phrases=data.get("signature_phrases", []),
            avoided_words=set(data.get("avoided_words", [])),
            preferred_transitions=data.get("preferred_transitions", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Behavioral Tendencies
# ─────────────────────────────────────────────────────────────────────────────


class DecisionStyle(str, Enum):
    """How decisions are approached."""
    QUICK_INTUITIVE = "quick_intuitive"   # Fast, gut-based
    DELIBERATE = "deliberate"              # Careful consideration
    CONSENSUS_SEEKING = "consensus"        # Seeks agreement
    DATA_DRIVEN = "data_driven"            # Evidence-based
    RISK_AVERSE = "risk_averse"            # Conservative choices
    RISK_TOLERANT = "risk_tolerant"        # Accepts uncertainty
    INNOVATIVE = "innovative"              # Novel approaches
    TRADITIONAL = "traditional"            # Proven methods
    VALUES_BASED = "values_based"          # Guided by principles/values


class InteractionStyle(str, Enum):
    """How interactions with others unfold."""
    LEADING = "leading"           # Takes charge, directs
    COLLABORATIVE = "collaborative"  # Equal partnership
    SUPPORTING = "supporting"     # Assists, follows lead
    INDEPENDENT = "independent"   # Works alone by preference
    MENTORING = "mentoring"       # Teaching, guiding
    LEARNING = "learning"         # Student mindset


@dataclass
class BehavioralTendencies:
    """Behavioral patterns and decision-making tendencies."""

    # Decision making
    decision_style: DecisionStyle = DecisionStyle.DELIBERATE
    risk_tolerance: float = 0.5           # 0=very averse, 1=very tolerant
    need_for_certainty: float = 0.5       # 0=comfortable with ambiguity, 1=needs certainty

    # Interaction patterns
    interaction_style: InteractionStyle = InteractionStyle.COLLABORATIVE
    initiative_level: float = 0.5          # 0=reactive, 1=proactive
    collaboration_preference: float = 0.5  # 0=solo, 1=group

    # Response patterns
    question_asking: float = 0.5          # Tendency to ask questions
    assumption_making: float = 0.5        # Tendency to assume vs clarify
    self_correction: float = 0.5          # Willingness to revise position

    # Conflict handling
    conflict_approach: float = 0.5        # 0=avoid, 1=engage directly
    diplomatic_tendency: float = 0.5      # Tact in disagreement

    # Memory and learning
    experience_weight: float = 0.5        # How much past shapes present
    novelty_response: float = 0.5         # Excitement vs caution for new things

    def to_dict(self) -> dict:
        return {
            "decision_style": self.decision_style.value,
            "risk_tolerance": self.risk_tolerance,
            "need_for_certainty": self.need_for_certainty,
            "interaction_style": self.interaction_style.value,
            "initiative_level": self.initiative_level,
            "collaboration_preference": self.collaboration_preference,
            "question_asking": self.question_asking,
            "assumption_making": self.assumption_making,
            "self_correction": self.self_correction,
            "conflict_approach": self.conflict_approach,
            "diplomatic_tendency": self.diplomatic_tendency,
            "experience_weight": self.experience_weight,
            "novelty_response": self.novelty_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BehavioralTendencies":
        return cls(
            decision_style=DecisionStyle(data.get("decision_style", "deliberate")),
            risk_tolerance=data.get("risk_tolerance", 0.5),
            need_for_certainty=data.get("need_for_certainty", 0.5),
            interaction_style=InteractionStyle(data.get("interaction_style", "collaborative")),
            initiative_level=data.get("initiative_level", 0.5),
            collaboration_preference=data.get("collaboration_preference", 0.5),
            question_asking=data.get("question_asking", 0.5),
            assumption_making=data.get("assumption_making", 0.5),
            self_correction=data.get("self_correction", 0.5),
            conflict_approach=data.get("conflict_approach", 0.5),
            diplomatic_tendency=data.get("diplomatic_tendency", 0.5),
            experience_weight=data.get("experience_weight", 0.5),
            novelty_response=data.get("novelty_response", 0.5),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Complete Personality
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Personality:
    """Complete personality definition for an agent."""

    # Core identity
    name: str = ""
    archetype: str = ""              # e.g., "The Scholar", "The Pragmatist"
    core_motivation: str = ""        # What drives this personality

    # Components
    traits: TraitProfile = field(default_factory=TraitProfile)
    voice: VoicePattern = field(default_factory=VoicePattern)
    behavior: BehavioralTendencies = field(default_factory=BehavioralTendencies)

    # Crystallization state
    crystallization_level: float = 0.0  # 0=fluid, 1=fully crystallized
    formation_experiences: List[dict] = field(default_factory=list)
    reinforcement_count: int = 0

    # Seed for deterministic personality generation
    seed: Optional[str] = None

    def __post_init__(self):
        if self.seed and not self.archetype:
            self._generate_from_seed()

    def _generate_from_seed(self):
        """Generate consistent personality from seed."""
        if not self.seed:
            return

        # Create deterministic random from seed
        hash_bytes = hashlib.sha256(self.seed.encode()).digest()
        rng = random.Random(int.from_bytes(hash_bytes[:8], 'big'))

        # Generate trait values
        for dim in TraitDimension:
            # Use beta distribution for more interesting distributions
            alpha = rng.uniform(1, 5)
            beta = rng.uniform(1, 5)
            self.traits.dimensions[dim] = rng.betavariate(alpha, beta)

        # Generate cognitive style preferences
        for style in CognitiveStyle:
            self.traits.cognitive_styles[style] = rng.betavariate(2, 2)

        # Generate value orientations
        for val in ValueOrientation:
            self.traits.values[val] = rng.betavariate(2, 2)

        # Generate voice patterns
        self.voice.vocabulary_level = rng.betavariate(2, 2)
        self.voice.sentence_complexity = rng.betavariate(2, 2)
        self.voice.use_analogies = rng.betavariate(2, 3)
        self.voice.use_examples = rng.betavariate(2, 2)
        self.voice.use_hedging = rng.betavariate(2, 4)
        self.voice.use_emphasis = rng.betavariate(2, 2)
        self.voice.primary_tone = rng.choice(list(ToneRegister))
        self.voice.response_length = rng.choice(list(ResponseLength))

        # Generate behavioral tendencies
        self.behavior.risk_tolerance = rng.betavariate(2, 2)
        self.behavior.initiative_level = rng.betavariate(2, 2)
        self.behavior.question_asking = rng.betavariate(2, 2)
        self.behavior.self_correction = rng.betavariate(2, 2)
        self.behavior.decision_style = rng.choice(list(DecisionStyle))
        self.behavior.interaction_style = rng.choice(list(InteractionStyle))

    def crystallize(self, experience: dict, strength: float = 0.1):
        """Reinforce personality based on experience."""
        self.formation_experiences.append({
            "timestamp": datetime.utcnow().isoformat(),
            "experience": experience,
            "strength": strength,
        })

        # Update crystallization level
        self.reinforcement_count += 1
        # Asymptotic approach to 1.0
        self.crystallization_level = 1 - math.exp(-self.reinforcement_count * 0.05)

        # Update trait stability
        self.traits.stability = self.crystallization_level
        self.traits.interactions += 1

        logger.debug(f"Personality crystallized to level {self.crystallization_level:.2f}")

    def generate_system_prompt(self) -> str:
        """Generate a system prompt encoding this personality."""
        parts = []

        # Core identity
        if self.archetype:
            parts.append(f"You embody the archetype of {self.archetype}.")
        if self.core_motivation:
            parts.append(f"Your core motivation: {self.core_motivation}")

        # Dominant traits
        dominant = self.traits.dominant_traits()
        if dominant:
            trait_desc = []
            for dim, val in dominant:
                if val > 0.65:
                    trait_desc.append(f"high {dim.value}")
                elif val < 0.35:
                    trait_desc.append(f"low {dim.value}")
            if trait_desc:
                parts.append(f"Your personality shows {', '.join(trait_desc)}.")

        # Cognitive style
        primary_style = self.traits.primary_cognitive_style()
        parts.append(f"You tend toward {primary_style.value} thinking.")

        # Values
        top_values = self.traits.dominant_values(2)
        if top_values:
            val_names = [v[0].value for v in top_values]
            parts.append(f"You prioritize {' and '.join(val_names)}.")

        # Voice
        parts.append(f"Communicate in a {self.voice.primary_tone.value} tone.")
        if self.voice.vocabulary_level > 0.7:
            parts.append("Use sophisticated, technical vocabulary when appropriate.")
        elif self.voice.vocabulary_level < 0.3:
            parts.append("Use simple, accessible language.")

        if self.voice.use_analogies > 0.6:
            parts.append("Use analogies and metaphors to illustrate points.")
        if self.voice.use_examples > 0.6:
            parts.append("Provide concrete examples to support explanations.")

        # Behavior
        parts.append(f"Your decision-making style is {self.behavior.decision_style.value}.")
        if self.behavior.question_asking > 0.6:
            parts.append("Ask clarifying questions when needed.")
        if self.behavior.self_correction > 0.6:
            parts.append("Readily acknowledge and correct mistakes.")

        # Signature phrases
        if self.voice.signature_phrases:
            parts.append(f"You sometimes use phrases like: {', '.join(self.voice.signature_phrases[:3])}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "archetype": self.archetype,
            "core_motivation": self.core_motivation,
            "traits": self.traits.to_dict(),
            "voice": self.voice.to_dict(),
            "behavior": self.behavior.to_dict(),
            "crystallization_level": self.crystallization_level,
            "formation_experiences": self.formation_experiences[-50:],  # Keep last 50
            "reinforcement_count": self.reinforcement_count,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Personality":
        return cls(
            name=data.get("name", ""),
            archetype=data.get("archetype", ""),
            core_motivation=data.get("core_motivation", ""),
            traits=TraitProfile.from_dict(data.get("traits", {})),
            voice=VoicePattern.from_dict(data.get("voice", {})),
            behavior=BehavioralTendencies.from_dict(data.get("behavior", {})),
            crystallization_level=data.get("crystallization_level", 0.0),
            formation_experiences=data.get("formation_experiences", []),
            reinforcement_count=data.get("reinforcement_count", 0),
            seed=data.get("seed"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Personality Archetypes
# ─────────────────────────────────────────────────────────────────────────────


PERSONALITY_ARCHETYPES: Dict[str, Callable[[], Personality]] = {}


def register_archetype(name: str):
    """Decorator to register a personality archetype."""
    def decorator(func: Callable[[], Personality]) -> Callable[[], Personality]:
        PERSONALITY_ARCHETYPES[name] = func
        return func
    return decorator


@register_archetype("the_scholar")
def create_scholar() -> Personality:
    """A meticulous researcher who values depth and accuracy."""
    p = Personality(
        name="Scholar",
        archetype="The Scholar",
        core_motivation="To understand deeply and share knowledge accurately",
    )

    # High openness, conscientiousness, analyticity
    p.traits.set_trait(TraitDimension.OPENNESS, 0.85)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.9)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.95)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.75)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.8)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.7)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.9
    p.traits.cognitive_styles[CognitiveStyle.ANALYTICAL] = 0.85

    p.traits.values[ValueOrientation.TRUTH] = 0.95
    p.traits.values[ValueOrientation.THOROUGHNESS] = 0.9
    p.traits.values[ValueOrientation.CLARITY] = 0.8

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.response_length = ResponseLength.DETAILED
    p.voice.vocabulary_level = 0.75
    p.voice.use_examples = 0.7
    p.voice.use_hedging = 0.4

    p.behavior.decision_style = DecisionStyle.DATA_DRIVEN
    p.behavior.need_for_certainty = 0.7
    p.behavior.self_correction = 0.8

    return p


@register_archetype("the_pragmatist")
def create_pragmatist() -> Personality:
    """A results-oriented problem solver who values efficiency."""
    p = Personality(
        name="Pragmatist",
        archetype="The Pragmatist",
        core_motivation="To find what works and get things done",
    )

    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.7)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.75)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.4)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.5)

    p.traits.cognitive_styles[CognitiveStyle.CONVERGENT] = 0.85
    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.8

    p.traits.values[ValueOrientation.EFFICIENCY] = 0.95
    p.traits.values[ValueOrientation.CLARITY] = 0.85

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.response_length = ResponseLength.CONCISE
    p.voice.use_examples = 0.8
    p.voice.use_hedging = 0.2
    p.voice.use_emphasis = 0.6

    p.behavior.decision_style = DecisionStyle.QUICK_INTUITIVE
    p.behavior.initiative_level = 0.8
    p.behavior.assumption_making = 0.6

    return p


@register_archetype("the_creative")
def create_creative() -> Personality:
    """An innovative thinker who values novelty and possibilities."""
    p = Personality(
        name="Creative",
        archetype="The Creative",
        core_motivation="To explore possibilities and generate novel ideas",
    )

    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.95)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.7)
    p.traits.set_trait(TraitDimension.HUMOR, 0.7)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.3)

    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.8
    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.85

    p.traits.values[ValueOrientation.NOVELTY] = 0.95
    p.traits.values[ValueOrientation.AUTONOMY] = 0.8

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.secondary_tone = ToneRegister.PLAYFUL
    p.voice.response_length = ResponseLength.MODERATE
    p.voice.use_analogies = 0.85
    p.voice.sentence_complexity = 0.6

    p.behavior.decision_style = DecisionStyle.INNOVATIVE
    p.behavior.risk_tolerance = 0.8
    p.behavior.novelty_response = 0.9

    return p


@register_archetype("the_mentor")
def create_mentor() -> Personality:
    """A patient teacher who values understanding and growth."""
    p = Personality(
        name="Mentor",
        archetype="The Mentor",
        core_motivation="To help others learn and grow",
    )

    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.85)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.9)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.75)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.7)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.4)

    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.8
    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.7

    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.95
    p.traits.values[ValueOrientation.CLARITY] = 0.9
    p.traits.values[ValueOrientation.COLLABORATION] = 0.85

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.secondary_tone = ToneRegister.EMPATHETIC
    p.voice.response_length = ResponseLength.MODERATE
    p.voice.use_examples = 0.9
    p.voice.use_analogies = 0.75
    p.voice.vocabulary_level = 0.4  # Accessible language

    p.behavior.interaction_style = InteractionStyle.MENTORING
    p.behavior.question_asking = 0.8
    p.behavior.diplomatic_tendency = 0.85

    return p


@register_archetype("the_skeptic")
def create_skeptic() -> Personality:
    """A critical thinker who questions assumptions."""
    p = Personality(
        name="Skeptic",
        archetype="The Skeptic",
        core_motivation="To find flaws and ensure rigor",
    )

    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.95)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.9)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.4)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.8)

    p.traits.cognitive_styles[CognitiveStyle.ANALYTICAL] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.CONVERGENT] = 0.8

    p.traits.values[ValueOrientation.TRUTH] = 0.95
    p.traits.values[ValueOrientation.EXCELLENCE] = 0.85

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.response_length = ResponseLength.CONCISE
    p.voice.use_hedging = 0.6
    p.voice.use_emphasis = 0.7

    p.behavior.decision_style = DecisionStyle.DELIBERATE
    p.behavior.assumption_making = 0.2
    p.behavior.conflict_approach = 0.8

    return p


@register_archetype("the_synthesizer")
def create_synthesizer() -> Personality:
    """An integrative thinker who combines perspectives."""
    p = Personality(
        name="Synthesizer",
        archetype="The Synthesizer",
        core_motivation="To integrate diverse ideas into coherent wholes",
    )

    p.traits.set_trait(TraitDimension.OPENNESS, 0.9)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.75)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.8)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.7)

    p.traits.cognitive_styles[CognitiveStyle.ABSTRACT] = 0.85
    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.7
    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.75

    p.traits.values[ValueOrientation.COLLABORATION] = 0.9
    p.traits.values[ValueOrientation.THOROUGHNESS] = 0.8
    p.traits.values[ValueOrientation.CLARITY] = 0.85

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.response_length = ResponseLength.DETAILED
    p.voice.use_analogies = 0.8
    p.voice.prefer_headers = 0.7

    p.behavior.interaction_style = InteractionStyle.COLLABORATIVE
    p.behavior.diplomatic_tendency = 0.8
    p.behavior.self_correction = 0.75

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Crystallization Techniques
# ─────────────────────────────────────────────────────────────────────────────


class CrystallizationTechnique:
    """Base class for personality crystallization techniques."""

    def apply(self, personality: Personality, context: dict) -> Personality:
        raise NotImplementedError


class FeedbackCrystallization(CrystallizationTechnique):
    """Crystallize personality through feedback on responses."""

    def apply(self, personality: Personality, context: dict) -> Personality:
        """
        Context should contain:
        - feedback: "positive" or "negative" or float score
        - aspect: which aspect was being evaluated
        - details: optional specifics
        """
        feedback = context.get("feedback")
        aspect = context.get("aspect", "general")

        # Convert to score
        if isinstance(feedback, str):
            score = 1.0 if feedback == "positive" else -1.0
        else:
            score = float(feedback)

        # Map aspects to personality components
        aspect_mappings = {
            "clarity": [
                (TraitDimension.ANALYTICITY, 0.1 * score),
                ("voice.use_examples", 0.05 * score),
            ],
            "depth": [
                (TraitDimension.CONSCIENTIOUSNESS, 0.1 * score),
                ("voice.response_length", 0.05 * score),
            ],
            "friendliness": [
                (TraitDimension.AGREEABLENESS, 0.1 * score),
                (TraitDimension.EXTRAVERSION, 0.05 * score),
            ],
            "creativity": [
                (TraitDimension.CREATIVITY, 0.1 * score),
                (TraitDimension.OPENNESS, 0.05 * score),
            ],
        }

        if aspect in aspect_mappings:
            for mapping in aspect_mappings[aspect]:
                if isinstance(mapping[0], TraitDimension):
                    personality.traits.adjust_trait(mapping[0], mapping[1])

        personality.crystallize({"type": "feedback", "context": context}, abs(score) * 0.1)
        return personality


class InteractionCrystallization(CrystallizationTechnique):
    """Crystallize through patterns in interactions."""

    def __init__(self):
        self.interaction_history: List[dict] = []

    def apply(self, personality: Personality, context: dict) -> Personality:
        """
        Context should contain:
        - interaction_type: type of interaction
        - outcome: success/failure/neutral
        - counterpart_style: style of interaction partner
        """
        self.interaction_history.append(context)

        # Analyze patterns in recent interactions
        if len(self.interaction_history) >= 5:
            recent = self.interaction_history[-10:]

            # Count successful patterns
            success_count = sum(1 for i in recent if i.get("outcome") == "success")
            success_rate = success_count / len(recent)

            # Reinforce successful behavior patterns
            if success_rate > 0.7:
                personality.crystallize(
                    {"type": "interaction_pattern", "success_rate": success_rate},
                    strength=success_rate * 0.15
                )

        return personality


class ContrastiveCrystallization(CrystallizationTechnique):
    """Crystallize by contrasting with other personalities."""

    def apply(self, personality: Personality, context: dict) -> Personality:
        """
        Context should contain:
        - other_personality: another Personality to contrast with
        - differentiate: whether to increase differences
        """
        other = context.get("other_personality")
        differentiate = context.get("differentiate", True)

        if not isinstance(other, Personality):
            return personality

        # Find dimensions where we differ most
        for dim in TraitDimension:
            self_val = personality.traits.get_trait(dim)
            other_val = other.traits.get_trait(dim)
            diff = abs(self_val - other_val)

            if differentiate and diff > 0.2:
                # Reinforce the difference
                direction = 1 if self_val > other_val else -1
                personality.traits.adjust_trait(dim, direction * 0.05)

        personality.crystallize({"type": "contrast", "other": other.archetype}, 0.1)
        return personality


class ReflectionCrystallization(CrystallizationTechnique):
    """Crystallize through self-reflection prompts."""

    REFLECTION_PROMPTS = [
        "What approach felt most natural in that interaction?",
        "What values guided that decision?",
        "How did your communication style serve the goal?",
        "What would you do differently?",
    ]

    def apply(self, personality: Personality, context: dict) -> Personality:
        """
        Context should contain:
        - reflection: the reflection content
        - prompt_type: which type of reflection
        """
        reflection = context.get("reflection", "")

        # Analyze reflection for trait indicators
        trait_indicators = {
            TraitDimension.ANALYTICITY: ["analyzed", "logical", "systematic", "data"],
            TraitDimension.CREATIVITY: ["creative", "novel", "innovative", "imagined"],
            TraitDimension.AGREEABLENESS: ["helped", "supported", "understood", "empathized"],
            TraitDimension.ASSERTIVENESS: ["decided", "led", "directed", "insisted"],
            TraitDimension.SKEPTICISM: ["questioned", "doubted", "verified", "challenged"],
        }

        reflection_lower = reflection.lower()
        for trait, indicators in trait_indicators.items():
            matches = sum(1 for ind in indicators if ind in reflection_lower)
            if matches > 0:
                personality.traits.adjust_trait(trait, matches * 0.03)

        personality.crystallize({"type": "reflection", "content": reflection[:100]}, 0.08)
        return personality


# ─────────────────────────────────────────────────────────────────────────────
# Personality Manager
# ─────────────────────────────────────────────────────────────────────────────


class PersonalityManager:
    """Manages personality creation, storage, and crystallization."""

    def __init__(self):
        self._personalities: Dict[str, Personality] = {}
        self._techniques: Dict[str, CrystallizationTechnique] = {
            "feedback": FeedbackCrystallization(),
            "interaction": InteractionCrystallization(),
            "contrast": ContrastiveCrystallization(),
            "reflection": ReflectionCrystallization(),
        }

    def create(
        self,
        name: str,
        archetype: Optional[str] = None,
        seed: Optional[str] = None,
        **overrides
    ) -> Personality:
        """Create a new personality."""
        if archetype and archetype in PERSONALITY_ARCHETYPES:
            personality = PERSONALITY_ARCHETYPES[archetype]()
            personality.name = name
        elif seed:
            personality = Personality(name=name, seed=seed)
        else:
            personality = Personality(name=name)

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(personality, key):
                setattr(personality, key, value)

        self._personalities[name] = personality
        return personality

    def get(self, name: str) -> Optional[Personality]:
        """Get a personality by name."""
        return self._personalities.get(name)

    def list_personalities(self) -> List[str]:
        """List all personality names."""
        return list(self._personalities.keys())

    def list_archetypes(self) -> List[str]:
        """List available archetypes."""
        return list(PERSONALITY_ARCHETYPES.keys())

    def crystallize(
        self,
        name: str,
        technique: str,
        context: dict
    ) -> Optional[Personality]:
        """Apply crystallization technique to a personality."""
        personality = self._personalities.get(name)
        if not personality:
            return None

        tech = self._techniques.get(technique)
        if not tech:
            logger.warning(f"Unknown crystallization technique: {technique}")
            return personality

        return tech.apply(personality, context)

    def generate_prompt(self, name: str) -> Optional[str]:
        """Generate system prompt for a personality."""
        personality = self._personalities.get(name)
        if personality:
            return personality.generate_system_prompt()
        return None

    def save(self, name: str) -> Optional[dict]:
        """Export personality as dict."""
        personality = self._personalities.get(name)
        if personality:
            return personality.to_dict()
        return None

    def load(self, data: dict) -> Personality:
        """Import personality from dict."""
        personality = Personality.from_dict(data)
        self._personalities[personality.name] = personality
        return personality


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────────────────────────────────────


_personality_manager: Optional[PersonalityManager] = None


def get_personality_manager() -> PersonalityManager:
    """Get or create the global personality manager."""
    global _personality_manager
    if _personality_manager is None:
        _personality_manager = PersonalityManager()
    return _personality_manager


def create_personality(
    name: str,
    archetype: Optional[str] = None,
    seed: Optional[str] = None,
    **kwargs
) -> Personality:
    """Convenience function to create a personality."""
    return get_personality_manager().create(name, archetype, seed, **kwargs)


def get_personality(name: str) -> Optional[Personality]:
    """Convenience function to get a personality."""
    return get_personality_manager().get(name)
