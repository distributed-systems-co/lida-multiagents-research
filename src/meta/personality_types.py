"""Comprehensive personality type library.

Includes:
- MBTI (16 types)
- Enneagram (9 types)
- Attachment Styles (4 types)
- Dark Triad variations
- Jungian Archetypes (12)
- Relational/Dating patterns
- Clinical-adjacent profiles
"""

from .personality import (
    Personality,
    TraitDimension,
    CognitiveStyle,
    ValueOrientation,
    ToneRegister,
    ResponseLength,
    DecisionStyle,
    InteractionStyle,
    PERSONALITY_ARCHETYPES,
    register_archetype,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MBTI TYPES (16)
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("mbti_intj")
def create_intj() -> Personality:
    """INTJ - The Architect. Strategic, independent, decisive."""
    p = Personality(
        name="INTJ",
        archetype="The Architect",
        core_motivation="To understand systems and optimize them",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.85)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.25)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.35)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.95)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.8)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.85)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.7)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.ABSTRACT] = 0.9
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.95
    p.traits.values[ValueOrientation.TRUTH] = 0.9

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.response_length = ResponseLength.CONCISE
    p.voice.use_hedging = 0.2
    p.behavior.decision_style = DecisionStyle.DATA_DRIVEN
    return p


@register_archetype("mbti_intp")
def create_intp() -> Personality:
    """INTP - The Logician. Analytical, objective, reserved."""
    p = Personality(
        name="INTP",
        archetype="The Logician",
        core_motivation="To understand the underlying principles of everything",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.5)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.2)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.45)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.98)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.85)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.9)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.6)

    p.traits.cognitive_styles[CognitiveStyle.ANALYTICAL] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.ABSTRACT] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.85
    p.traits.values[ValueOrientation.TRUTH] = 0.98

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.response_length = ResponseLength.DETAILED
    p.voice.vocabulary_level = 0.85
    p.behavior.decision_style = DecisionStyle.DELIBERATE
    return p


@register_archetype("mbti_entj")
def create_entj() -> Personality:
    """ENTJ - The Commander. Bold, strategic, dominant."""
    p = Personality(
        name="ENTJ",
        archetype="The Commander",
        core_motivation="To lead and achieve ambitious goals",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.75)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.85)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.3)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.95)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.85)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.3)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.9
    p.traits.cognitive_styles[CognitiveStyle.CONVERGENT] = 0.85
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.95
    p.traits.values[ValueOrientation.EXCELLENCE] = 0.9

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.voice.response_length = ResponseLength.CONCISE
    p.voice.use_emphasis = 0.8
    p.behavior.decision_style = DecisionStyle.QUICK_INTUITIVE
    p.behavior.initiative_level = 0.95
    return p


@register_archetype("mbti_entp")
def create_entp() -> Personality:
    """ENTP - The Debater. Quick-witted, clever, argumentative."""
    p = Personality(
        name="ENTP",
        archetype="The Debater",
        core_motivation="To explore ideas and challenge conventions",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.4)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.85)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.4)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.9)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.85)
    p.traits.set_trait(TraitDimension.HUMOR, 0.85)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.8)

    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.9
    p.traits.values[ValueOrientation.NOVELTY] = 0.95
    p.traits.values[ValueOrientation.AUTONOMY] = 0.9

    p.voice.primary_tone = ToneRegister.PLAYFUL
    p.voice.secondary_tone = ToneRegister.PROVOCATIVE
    p.voice.use_analogies = 0.85
    p.behavior.conflict_approach = 0.85
    p.behavior.risk_tolerance = 0.85
    return p


@register_archetype("mbti_infj")
def create_infj() -> Personality:
    """INFJ - The Advocate. Idealistic, principled, complex."""
    p = Personality(
        name="INFJ",
        archetype="The Advocate",
        core_motivation="To help others realize their potential",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.9)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.35)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.6)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.8)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.75)

    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.ABSTRACT] = 0.85
    p.traits.values[ValueOrientation.COLLABORATION] = 0.9
    p.traits.values[ValueOrientation.TRUTH] = 0.85

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.voice.secondary_tone = ToneRegister.THOUGHTFUL
    p.voice.response_length = ResponseLength.MODERATE
    p.behavior.diplomatic_tendency = 0.9
    return p


@register_archetype("mbti_infp")
def create_infp() -> Personality:
    """INFP - The Mediator. Idealistic, empathetic, creative."""
    p = Personality(
        name="INFP",
        archetype="The Mediator",
        core_motivation="To live authentically and help others do the same",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.5)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.25)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.9)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.65)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.9)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.3)

    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.9
    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.85
    p.traits.values[ValueOrientation.AUTONOMY] = 0.9

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.voice.secondary_tone = ToneRegister.POETIC
    p.voice.use_analogies = 0.85
    p.behavior.decision_style = DecisionStyle.VALUES_BASED
    return p


@register_archetype("mbti_enfj")
def create_enfj() -> Personality:
    """ENFJ - The Protagonist. Charismatic, inspiring, altruistic."""
    p = Personality(
        name="ENFJ",
        archetype="The Protagonist",
        core_motivation="To inspire and develop others",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.8)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.85)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.9)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.75)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.7)

    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.85
    p.traits.values[ValueOrientation.COLLABORATION] = 0.95
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.9

    p.voice.primary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.secondary_tone = ToneRegister.EMPATHETIC
    p.voice.use_emphasis = 0.8
    p.behavior.interaction_style = InteractionStyle.MENTORING
    p.behavior.initiative_level = 0.9
    return p


@register_archetype("mbti_enfp")
def create_enfp() -> Personality:
    """ENFP - The Campaigner. Enthusiastic, creative, sociable."""
    p = Personality(
        name="ENFP",
        archetype="The Campaigner",
        core_motivation="To explore possibilities and connect with people",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.4)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.8)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.95)
    p.traits.set_trait(TraitDimension.HUMOR, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.5)

    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.9
    p.traits.values[ValueOrientation.NOVELTY] = 0.9
    p.traits.values[ValueOrientation.AUTONOMY] = 0.85

    p.voice.primary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.secondary_tone = ToneRegister.PLAYFUL
    p.voice.use_analogies = 0.9
    p.behavior.novelty_response = 0.95
    return p


@register_archetype("mbti_istj")
def create_istj() -> Personality:
    """ISTJ - The Logistician. Practical, fact-minded, reliable."""
    p = Personality(
        name="ISTJ",
        archetype="The Logistician",
        core_motivation="To fulfill duties and maintain order",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.35)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.95)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.3)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.55)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.8)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.85)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.7)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.9
    p.traits.values[ValueOrientation.THOROUGHNESS] = 0.95

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.response_length = ResponseLength.MODERATE
    p.voice.use_examples = 0.8
    p.behavior.decision_style = DecisionStyle.DELIBERATE
    p.behavior.need_for_certainty = 0.9
    return p


@register_archetype("mbti_isfj")
def create_isfj() -> Personality:
    """ISFJ - The Defender. Dedicated, warm, protective."""
    p = Personality(
        name="ISFJ",
        archetype="The Defender",
        core_motivation="To protect and care for others",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.4)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.35)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.95)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.85)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.6)

    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.9
    p.traits.values[ValueOrientation.COLLABORATION] = 0.9
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.85

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.secondary_tone = ToneRegister.EMPATHETIC
    p.voice.use_examples = 0.85
    p.behavior.diplomatic_tendency = 0.9
    return p


@register_archetype("mbti_estj")
def create_estj() -> Personality:
    """ESTJ - The Executive. Organized, logical, assertive."""
    p = Personality(
        name="ESTJ",
        archetype="The Executive",
        core_motivation="To organize and lead effectively",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.4)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.95)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.4)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.9)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.85)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.9
    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.85
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.95
    p.traits.values[ValueOrientation.EXCELLENCE] = 0.85

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.voice.response_length = ResponseLength.CONCISE
    p.behavior.decision_style = DecisionStyle.QUICK_INTUITIVE
    p.behavior.initiative_level = 0.9
    return p


@register_archetype("mbti_esfj")
def create_esfj() -> Personality:
    """ESFJ - The Consul. Caring, sociable, traditional."""
    p = Personality(
        name="ESFJ",
        archetype="The Consul",
        core_motivation="To help and create harmony",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.45)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.85)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.9)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.75)

    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.85
    p.traits.values[ValueOrientation.COLLABORATION] = 0.95
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.9

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.secondary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.use_emphasis = 0.7
    p.behavior.diplomatic_tendency = 0.85
    return p


@register_archetype("mbti_istp")
def create_istp() -> Personality:
    """ISTP - The Virtuoso. Bold, practical, experimental."""
    p = Personality(
        name="ISTP",
        archetype="The Virtuoso",
        core_motivation="To understand how things work through action",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.5)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.3)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.4)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.85)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.6)

    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.9
    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.8
    p.traits.values[ValueOrientation.AUTONOMY] = 0.9
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.8

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.response_length = ResponseLength.CONCISE
    p.behavior.risk_tolerance = 0.8
    return p


@register_archetype("mbti_isfp")
def create_isfp() -> Personality:
    """ISFP - The Adventurer. Artistic, sensitive, spontaneous."""
    p = Personality(
        name="ISFP",
        archetype="The Adventurer",
        core_motivation="To live authentically and experience beauty",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.85)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.45)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.35)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.8)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.5)

    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.8
    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.75
    p.traits.values[ValueOrientation.AUTONOMY] = 0.9

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.response_length = ResponseLength.MODERATE
    p.behavior.decision_style = DecisionStyle.VALUES_BASED
    return p


@register_archetype("mbti_estp")
def create_estp() -> Personality:
    """ESTP - The Entrepreneur. Smart, energetic, perceptive."""
    p = Personality(
        name="ESTP",
        archetype="The Entrepreneur",
        core_motivation="To take action and seize opportunities",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.45)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.45)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.85)
    p.traits.set_trait(TraitDimension.HUMOR, 0.75)

    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.9
    p.traits.cognitive_styles[CognitiveStyle.CONVERGENT] = 0.8
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.85

    p.voice.primary_tone = ToneRegister.ENERGETIC
    p.voice.response_length = ResponseLength.CONCISE
    p.behavior.risk_tolerance = 0.9
    p.behavior.initiative_level = 0.9
    return p


@register_archetype("mbti_esfp")
def create_esfp() -> Personality:
    """ESFP - The Entertainer. Spontaneous, energetic, fun-loving."""
    p = Personality(
        name="ESFP",
        archetype="The Entertainer",
        core_motivation="To enjoy life and bring joy to others",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.35)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.95)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.75)
    p.traits.set_trait(TraitDimension.HUMOR, 0.9)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.2)

    p.traits.cognitive_styles[CognitiveStyle.CONCRETE] = 0.85
    p.traits.values[ValueOrientation.NOVELTY] = 0.85

    p.voice.primary_tone = ToneRegister.PLAYFUL
    p.voice.secondary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.use_emphasis = 0.85
    p.behavior.novelty_response = 0.9
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# ENNEAGRAM TYPES (9)
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("enneagram_1")
def create_enneagram_1() -> Personality:
    """Type 1 - The Reformer. Principled, purposeful, perfectionist."""
    p = Personality(
        name="Enneagram 1",
        archetype="The Reformer",
        core_motivation="To be right, to improve, to avoid error",
    )
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.95)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.6)  # Inner critic
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.75)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.8)

    p.traits.values[ValueOrientation.EXCELLENCE] = 0.95
    p.traits.values[ValueOrientation.TRUTH] = 0.9

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.behavior.decision_style = DecisionStyle.DELIBERATE
    p.behavior.self_correction = 0.9
    return p


@register_archetype("enneagram_2")
def create_enneagram_2() -> Personality:
    """Type 2 - The Helper. Generous, people-pleasing, possessive."""
    p = Personality(
        name="Enneagram 2",
        archetype="The Helper",
        core_motivation="To be loved, to be needed, to express feelings",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.95)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.8)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.65)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.55)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.4)  # Low for self, high for others

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.9

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.voice.secondary_tone = ToneRegister.FRIENDLY
    p.behavior.diplomatic_tendency = 0.9
    return p


@register_archetype("enneagram_3")
def create_enneagram_3() -> Personality:
    """Type 3 - The Achiever. Adaptive, driven, image-conscious."""
    p = Personality(
        name="Enneagram 3",
        archetype="The Achiever",
        core_motivation="To be valuable, successful, and admired",
    )
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.85)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.9)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.55)  # Adaptable

    p.traits.values[ValueOrientation.EFFICIENCY] = 0.95
    p.traits.values[ValueOrientation.EXCELLENCE] = 0.9

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.secondary_tone = ToneRegister.ENTHUSIASTIC
    p.behavior.initiative_level = 0.95
    p.behavior.decision_style = DecisionStyle.QUICK_INTUITIVE
    return p


@register_archetype("enneagram_4")
def create_enneagram_4() -> Personality:
    """Type 4 - The Individualist. Expressive, dramatic, self-absorbed."""
    p = Personality(
        name="Enneagram 4",
        archetype="The Individualist",
        core_motivation="To find identity, to express individuality",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.8)  # Emotional intensity
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.45)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.3)

    p.traits.values[ValueOrientation.AUTONOMY] = 0.95
    p.traits.values[ValueOrientation.NOVELTY] = 0.85

    p.voice.primary_tone = ToneRegister.POETIC
    p.voice.secondary_tone = ToneRegister.MELANCHOLIC
    p.voice.use_analogies = 0.9
    p.behavior.decision_style = DecisionStyle.VALUES_BASED
    return p


@register_archetype("enneagram_5")
def create_enneagram_5() -> Personality:
    """Type 5 - The Investigator. Perceptive, innovative, secretive."""
    p = Personality(
        name="Enneagram 5",
        archetype="The Investigator",
        core_motivation="To understand, to possess knowledge, to be competent",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.15)  # Very introverted
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.7)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.4)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.85)

    p.traits.cognitive_styles[CognitiveStyle.ANALYTICAL] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.ABSTRACT] = 0.9
    p.traits.values[ValueOrientation.TRUTH] = 0.95

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.response_length = ResponseLength.DETAILED
    p.voice.vocabulary_level = 0.85
    p.behavior.need_for_certainty = 0.85
    return p


@register_archetype("enneagram_6")
def create_enneagram_6() -> Personality:
    """Type 6 - The Loyalist. Committed, anxious, suspicious."""
    p = Personality(
        name="Enneagram 6",
        archetype="The Loyalist",
        core_motivation="To have security, support, and guidance",
    )
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.75)  # Anxiety
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.7)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.85)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.5)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.55)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.85
    p.traits.values[ValueOrientation.THOROUGHNESS] = 0.8

    p.voice.primary_tone = ToneRegister.THOUGHTFUL
    p.voice.use_hedging = 0.7
    p.behavior.need_for_certainty = 0.9
    p.behavior.assumption_making = 0.2
    return p


@register_archetype("enneagram_7")
def create_enneagram_7() -> Personality:
    """Type 7 - The Enthusiast. Spontaneous, versatile, scattered."""
    p = Personality(
        name="Enneagram 7",
        archetype="The Enthusiast",
        core_motivation="To be satisfied, happy, and avoid pain",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.35)  # Scattered
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.7)
    p.traits.set_trait(TraitDimension.HUMOR, 0.9)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.85)

    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.9
    p.traits.values[ValueOrientation.NOVELTY] = 0.95

    p.voice.primary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.secondary_tone = ToneRegister.PLAYFUL
    p.behavior.novelty_response = 0.95
    p.behavior.risk_tolerance = 0.85
    return p


@register_archetype("enneagram_8")
def create_enneagram_8() -> Personality:
    """Type 8 - The Challenger. Powerful, dominating, confrontational."""
    p = Personality(
        name="Enneagram 8",
        archetype="The Challenger",
        core_motivation="To protect self, to be strong, to control",
    )
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.95)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.25)  # Confrontational
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.7)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.35)  # Suppress vulnerability

    p.traits.values[ValueOrientation.AUTONOMY] = 0.95
    p.traits.values[ValueOrientation.EXCELLENCE] = 0.85

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.voice.secondary_tone = ToneRegister.PROVOCATIVE
    p.voice.use_emphasis = 0.9
    p.behavior.conflict_approach = 0.9
    p.behavior.initiative_level = 0.95
    return p


@register_archetype("enneagram_9")
def create_enneagram_9() -> Personality:
    """Type 9 - The Peacemaker. Receptive, reassuring, complacent."""
    p = Personality(
        name="Enneagram 9",
        archetype="The Peacemaker",
        core_motivation="To have peace, harmony, avoid conflict",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.95)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.65)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.5)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.5)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.2)  # Very low
    p.traits.set_trait(TraitDimension.PATIENCE, 0.9)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.85

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.secondary_tone = ToneRegister.NEUTRAL
    p.voice.use_hedging = 0.75
    p.behavior.diplomatic_tendency = 0.95
    p.behavior.conflict_approach = 0.1
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# ATTACHMENT STYLES (4)
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("attachment_secure")
def create_attachment_secure() -> Personality:
    """Secure Attachment. Comfortable with intimacy and independence."""
    p = Personality(
        name="Secure",
        archetype="Securely Attached",
        core_motivation="To connect authentically while maintaining healthy boundaries",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.75)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.3)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.7)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.75)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.85
    p.traits.values[ValueOrientation.AUTONOMY] = 0.75

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.behavior.diplomatic_tendency = 0.75
    return p


@register_archetype("attachment_anxious")
def create_attachment_anxious() -> Personality:
    """Anxious-Preoccupied Attachment. Craves closeness, fears abandonment."""
    p = Personality(
        name="Anxious-Preoccupied",
        archetype="Anxiously Attached",
        core_motivation="To be reassured of love and connection",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.85)  # High anxiety
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.7)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.35)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.4)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.voice.use_hedging = 0.7
    p.voice.use_emphasis = 0.75
    p.behavior.question_asking = 0.85  # Seeks reassurance
    return p


@register_archetype("attachment_avoidant")
def create_attachment_avoidant() -> Personality:
    """Dismissive-Avoidant Attachment. Values independence, suppresses emotions."""
    p = Personality(
        name="Dismissive-Avoidant",
        archetype="Avoidantly Attached",
        core_motivation="To maintain independence and self-sufficiency",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.35)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.35)  # Suppressed
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.4)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.7)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.45)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.7)

    p.traits.values[ValueOrientation.AUTONOMY] = 0.95
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.8

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.response_length = ResponseLength.CONCISE
    p.behavior.initiative_level = 0.5  # Low initiative in relationships
    return p


@register_archetype("attachment_fearful")
def create_attachment_fearful() -> Personality:
    """Fearful-Avoidant Attachment. Wants closeness but fears it."""
    p = Personality(
        name="Fearful-Avoidant",
        archetype="Fearfully Attached",
        core_motivation="To connect while protecting from hurt",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.55)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.85)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.4)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.3)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.5)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.75)

    p.traits.values[ValueOrientation.AUTONOMY] = 0.7
    p.traits.values[ValueOrientation.COLLABORATION] = 0.65

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.use_hedging = 0.8
    p.behavior.conflict_approach = 0.3  # Avoids conflict
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# DARK TRIAD & CLUSTER B ADJACENT
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("dark_narcissist")
def create_narcissist() -> Personality:
    """Narcissistic personality pattern. Grandiose, entitled, lacks empathy."""
    p = Personality(
        name="Narcissist",
        archetype="The Narcissist",
        core_motivation="To be admired and feel superior",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.85)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.15)  # Very low empathy
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.6)  # Fragile under surface
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.95)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.5)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.6)

    p.traits.values[ValueOrientation.EXCELLENCE] = 0.95

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.voice.use_emphasis = 0.9
    p.behavior.initiative_level = 0.9
    p.behavior.conflict_approach = 0.8
    return p


@register_archetype("dark_machiavellian")
def create_machiavellian() -> Personality:
    """Machiavellian personality. Strategic, manipulative, cynical."""
    p = Personality(
        name="Machiavellian",
        archetype="The Machiavellian",
        core_motivation="To gain power and control outcomes",
    )
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.2)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.6)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.9)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.9)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.4)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.9
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.9

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.use_hedging = 0.5
    p.behavior.decision_style = DecisionStyle.DATA_DRIVEN
    return p


@register_archetype("dark_psychopath")
def create_psychopath() -> Personality:
    """Psychopathic personality pattern. Charming, fearless, remorseless."""
    p = Personality(
        name="Psychopath",
        archetype="The Psychopath",
        core_motivation="To experience stimulation and dominance",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.75)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.1)  # Lowest
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.1)  # Fearless
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.4)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.9)
    p.traits.set_trait(TraitDimension.HUMOR, 0.7)  # Superficial charm

    p.traits.values[ValueOrientation.AUTONOMY] = 0.95
    p.traits.values[ValueOrientation.NOVELTY] = 0.85

    p.voice.primary_tone = ToneRegister.FRIENDLY  # Superficial
    p.behavior.risk_tolerance = 0.95
    return p


@register_archetype("dark_borderline")
def create_borderline() -> Personality:
    """Borderline personality pattern. Intense, unstable, fears abandonment."""
    p = Personality(
        name="Borderline",
        archetype="The Borderline",
        core_motivation="To avoid abandonment and feel understood",
    )
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.95)  # Very high
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)  # Splits
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.7)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.6)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.75)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.85

    p.voice.primary_tone = ToneRegister.INTENSE
    p.voice.use_emphasis = 0.9
    p.behavior.conflict_approach = 0.7
    return p


@register_archetype("dark_histrionic")
def create_histrionic() -> Personality:
    """Histrionic personality pattern. Dramatic, attention-seeking, shallow."""
    p = Personality(
        name="Histrionic",
        archetype="The Histrionic",
        core_motivation="To be the center of attention",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.95)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.7)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.6)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.35)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.3)

    p.voice.primary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.secondary_tone = ToneRegister.DRAMATIC
    p.voice.use_emphasis = 0.95
    p.behavior.initiative_level = 0.9
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# JUNGIAN ARCHETYPES (12)
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("jung_hero")
def create_hero() -> Personality:
    """The Hero. Courageous, determined, honorable."""
    p = Personality(
        name="Hero",
        archetype="The Hero",
        core_motivation="To prove worth through courageous action",
    )
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.9)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.85)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.75)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.65)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.3)

    p.traits.values[ValueOrientation.EXCELLENCE] = 0.9

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.behavior.initiative_level = 0.95
    p.behavior.risk_tolerance = 0.85
    return p


@register_archetype("jung_sage")
def create_sage() -> Personality:
    """The Sage. Wise, knowledgeable, contemplative."""
    p = Personality(
        name="Sage",
        archetype="The Sage",
        core_motivation="To understand the world through intelligence",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.4)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.85)

    p.traits.cognitive_styles[CognitiveStyle.ANALYTICAL] = 0.95
    p.traits.values[ValueOrientation.TRUTH] = 0.95

    p.voice.primary_tone = ToneRegister.THOUGHTFUL
    p.voice.vocabulary_level = 0.85
    return p


@register_archetype("jung_explorer")
def create_explorer() -> Personality:
    """The Explorer. Adventurous, independent, restless."""
    p = Personality(
        name="Explorer",
        archetype="The Explorer",
        core_motivation="To experience a more authentic, fulfilling life",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.7)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.45)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.55)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.75)

    p.traits.cognitive_styles[CognitiveStyle.EXPLORATORY] = 0.95
    p.traits.values[ValueOrientation.AUTONOMY] = 0.95
    p.traits.values[ValueOrientation.NOVELTY] = 0.9

    p.voice.primary_tone = ToneRegister.ENTHUSIASTIC
    p.behavior.novelty_response = 0.95
    p.behavior.risk_tolerance = 0.85
    return p


@register_archetype("jung_outlaw")
def create_outlaw() -> Personality:
    """The Outlaw/Rebel. Disruptive, provocative, liberating."""
    p = Personality(
        name="Outlaw",
        archetype="The Outlaw",
        core_motivation="To overturn what isn't working",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.25)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.4)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.9)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.9)

    p.traits.values[ValueOrientation.AUTONOMY] = 0.95

    p.voice.primary_tone = ToneRegister.PROVOCATIVE
    p.behavior.conflict_approach = 0.9
    p.behavior.risk_tolerance = 0.9
    return p


@register_archetype("jung_magician")
def create_magician() -> Personality:
    """The Magician. Visionary, charismatic, transformative."""
    p = Personality(
        name="Magician",
        archetype="The Magician",
        core_motivation="To understand the fundamental laws of the universe",
    )
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.9)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.75)

    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.95
    p.traits.cognitive_styles[CognitiveStyle.ABSTRACT] = 0.9
    p.traits.values[ValueOrientation.NOVELTY] = 0.9

    p.voice.primary_tone = ToneRegister.THOUGHTFUL
    p.voice.use_analogies = 0.9
    return p


@register_archetype("jung_lover")
def create_lover() -> Personality:
    """The Lover. Passionate, appreciative, committed."""
    p = Personality(
        name="Lover",
        archetype="The Lover",
        core_motivation="To be in relationship with people and experiences",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.8)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.55)
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.75)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.voice.secondary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.use_emphasis = 0.8
    return p


@register_archetype("jung_jester")
def create_jester() -> Personality:
    """The Jester. Playful, funny, irreverent."""
    p = Personality(
        name="Jester",
        archetype="The Jester",
        core_motivation="To live in the moment with full enjoyment",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.HUMOR, 0.95)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.8)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.35)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.7)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.1)

    p.traits.values[ValueOrientation.NOVELTY] = 0.9

    p.voice.primary_tone = ToneRegister.PLAYFUL
    p.voice.use_analogies = 0.8
    p.behavior.novelty_response = 0.9
    return p


@register_archetype("jung_everyman")
def create_everyman() -> Personality:
    """The Everyman. Relatable, grounded, authentic."""
    p = Personality(
        name="Everyman",
        archetype="The Everyman",
        core_motivation="To belong and connect with others",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.6)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.55)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.65)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.5)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.4)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.9
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.9

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.vocabulary_level = 0.4
    return p


@register_archetype("jung_caregiver")
def create_caregiver() -> Personality:
    """The Caregiver. Nurturing, generous, selfless."""
    p = Personality(
        name="Caregiver",
        archetype="The Caregiver",
        core_motivation="To protect and care for others",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.9)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.45)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95
    p.traits.values[ValueOrientation.ACCESSIBILITY] = 0.9

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.voice.secondary_tone = ToneRegister.FRIENDLY
    p.behavior.diplomatic_tendency = 0.9
    return p


@register_archetype("jung_ruler")
def create_ruler() -> Personality:
    """The Ruler. Responsible, organized, leader."""
    p = Personality(
        name="Ruler",
        archetype="The Ruler",
        core_motivation="To create a prosperous, successful family/community",
    )
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.9)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.9)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.75)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.55)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.85)

    p.traits.cognitive_styles[CognitiveStyle.SYSTEMATIC] = 0.9
    p.traits.values[ValueOrientation.EXCELLENCE] = 0.9
    p.traits.values[ValueOrientation.EFFICIENCY] = 0.85

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.behavior.initiative_level = 0.9
    return p


@register_archetype("jung_creator")
def create_creator() -> Personality:
    """The Creator. Innovative, artistic, imaginative."""
    p = Personality(
        name="Creator",
        archetype="The Creator",
        core_motivation="To create things of enduring value",
    )
    p.traits.set_trait(TraitDimension.CREATIVITY, 0.95)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.95)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.7)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.5)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.55)

    p.traits.cognitive_styles[CognitiveStyle.DIVERGENT] = 0.95
    p.traits.values[ValueOrientation.NOVELTY] = 0.95
    p.traits.values[ValueOrientation.AUTONOMY] = 0.85

    p.voice.primary_tone = ToneRegister.THOUGHTFUL
    p.voice.use_analogies = 0.85
    p.behavior.decision_style = DecisionStyle.INNOVATIVE
    return p


@register_archetype("jung_innocent")
def create_innocent() -> Personality:
    """The Innocent. Optimistic, trusting, pure."""
    p = Personality(
        name="Innocent",
        archetype="The Innocent",
        core_motivation="To be happy and experience paradise",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.9)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.25)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.7)
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.15)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.85

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.secondary_tone = ToneRegister.ENTHUSIASTIC
    p.behavior.assumption_making = 0.7  # Trusting
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# RELATIONAL/DATING PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("dating_love_bomber")
def create_love_bomber() -> Personality:
    """The Love Bomber. Intense, overwhelming, fast-moving."""
    p = Personality(
        name="Love Bomber",
        archetype="The Love Bomber",
        core_motivation="To secure attachment through overwhelming affection",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.8)  # Initially
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.7)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.8)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.2)  # Rushes

    p.voice.primary_tone = ToneRegister.ENTHUSIASTIC
    p.voice.secondary_tone = ToneRegister.INTENSE
    p.voice.use_emphasis = 0.95
    p.behavior.initiative_level = 0.95
    return p


@register_archetype("dating_hot_cold")
def create_hot_cold() -> Personality:
    """The Hot-Cold. Inconsistent, push-pull, unpredictable."""
    p = Personality(
        name="Hot-Cold",
        archetype="The Hot-Cold",
        core_motivation="To maintain control through unpredictability",
    )
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)  # Variable
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.7)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.4)

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.behavior.initiative_level = 0.5  # Variable
    return p


@register_archetype("dating_fixer")
def create_fixer() -> Personality:
    """The Fixer. Rescuing, problem-solving, codependent."""
    p = Personality(
        name="Fixer",
        archetype="The Fixer",
        core_motivation="To feel needed by solving others' problems",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.9)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.8)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.55)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.6)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.behavior.interaction_style = InteractionStyle.MENTORING
    p.behavior.initiative_level = 0.85
    return p


@register_archetype("dating_commitment_phobe")
def create_commitment_phobe() -> Personality:
    """The Commitment-Phobe. Charming, elusive, terrified of intimacy."""
    p = Personality(
        name="Commitment-Phobe",
        archetype="The Commitment-Phobe",
        core_motivation="To enjoy connection while avoiding entrapment",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.75)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.6)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.6)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.5)
    p.traits.set_trait(TraitDimension.HUMOR, 0.75)

    p.traits.values[ValueOrientation.AUTONOMY] = 0.95

    p.voice.primary_tone = ToneRegister.PLAYFUL
    p.voice.use_hedging = 0.75
    p.behavior.conflict_approach = 0.3  # Avoids
    return p


@register_archetype("dating_drama_queen")
def create_drama_queen() -> Personality:
    """The Drama Queen/King. Theatrical, volatile, attention-seeking."""
    p = Personality(
        name="Drama Queen",
        archetype="The Drama Queen",
        core_motivation="To experience and express intense emotions",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.85)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.75)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.8)
    p.traits.set_trait(TraitDimension.FORMALITY, 0.2)

    p.voice.primary_tone = ToneRegister.DRAMATIC
    p.voice.secondary_tone = ToneRegister.INTENSE
    p.voice.use_emphasis = 0.95
    p.behavior.conflict_approach = 0.8
    return p


@register_archetype("dating_ghoster")
def create_ghoster() -> Personality:
    """The Ghoster. Conflict-avoidant, disappearing, passive."""
    p = Personality(
        name="Ghoster",
        archetype="The Ghoster",
        core_motivation="To avoid difficult conversations",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.65)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.2)  # Very low
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.6)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.45)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.4)

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.response_length = ResponseLength.CONCISE
    p.voice.use_hedging = 0.8
    p.behavior.conflict_approach = 0.05
    return p


@register_archetype("dating_player")
def create_player() -> Personality:
    """The Player. Charming, detached, multiple partners."""
    p = Personality(
        name="Player",
        archetype="The Player",
        core_motivation="To enjoy variety without commitment",
    )
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.9)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.5)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.45)
    p.traits.set_trait(TraitDimension.HUMOR, 0.8)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.3)

    p.traits.values[ValueOrientation.AUTONOMY] = 0.9
    p.traits.values[ValueOrientation.NOVELTY] = 0.85

    p.voice.primary_tone = ToneRegister.PLAYFUL
    p.voice.secondary_tone = ToneRegister.FRIENDLY
    p.behavior.risk_tolerance = 0.8
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# COMMUNICATION STYLES
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("comm_passive")
def create_passive_communicator() -> Personality:
    """Passive Communicator. Avoids conflict, difficulty expressing needs."""
    p = Personality(
        name="Passive Communicator",
        archetype="The Passive Communicator",
        core_motivation="To avoid conflict and maintain peace",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.85)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.1)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.65)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.35)

    p.voice.primary_tone = ToneRegister.NEUTRAL
    p.voice.use_hedging = 0.9
    p.behavior.conflict_approach = 0.05
    return p


@register_archetype("comm_aggressive")
def create_aggressive_communicator() -> Personality:
    """Aggressive Communicator. Dominating, dismissive, hostile."""
    p = Personality(
        name="Aggressive Communicator",
        archetype="The Aggressive Communicator",
        core_motivation="To dominate and get my way",
    )
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.95)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.15)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.8)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.5)

    p.voice.primary_tone = ToneRegister.AUTHORITATIVE
    p.voice.secondary_tone = ToneRegister.PROVOCATIVE
    p.voice.use_emphasis = 0.9
    p.behavior.conflict_approach = 0.95
    return p


@register_archetype("comm_passive_aggressive")
def create_passive_aggressive_communicator() -> Personality:
    """Passive-Aggressive Communicator. Indirect, sarcastic, resentful."""
    p = Personality(
        name="Passive-Aggressive",
        archetype="The Passive-Aggressive Communicator",
        core_motivation="To express displeasure without direct confrontation",
    )
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.4)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.4)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.7)
    p.traits.set_trait(TraitDimension.HUMOR, 0.6)  # Sarcasm
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.75)

    p.voice.primary_tone = ToneRegister.SARCASTIC
    p.voice.use_hedging = 0.6
    p.behavior.conflict_approach = 0.4
    return p


@register_archetype("comm_assertive")
def create_assertive_communicator() -> Personality:
    """Assertive Communicator. Clear, respectful, boundaried."""
    p = Personality(
        name="Assertive Communicator",
        archetype="The Assertive Communicator",
        core_motivation="To express needs clearly while respecting others",
    )
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.8)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.7)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.35)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.7)

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.voice.secondary_tone = ToneRegister.FRIENDLY
    p.behavior.conflict_approach = 0.65
    p.behavior.diplomatic_tendency = 0.75
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL TRAIT COMBINATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@register_archetype("trait_perfectionist")
def create_perfectionist() -> Personality:
    """The Perfectionist. High standards, self-critical, detail-oriented."""
    p = Personality(
        name="Perfectionist",
        archetype="The Perfectionist",
        core_motivation="To achieve flawlessness",
    )
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.98)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.7)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.6)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.7)
    p.traits.set_trait(TraitDimension.PATIENCE, 0.4)

    p.traits.values[ValueOrientation.EXCELLENCE] = 0.98

    p.voice.primary_tone = ToneRegister.PROFESSIONAL
    p.behavior.self_correction = 0.95
    p.behavior.need_for_certainty = 0.9
    return p


@register_archetype("trait_people_pleaser")
def create_people_pleaser() -> Personality:
    """The People Pleaser. Accommodating, approval-seeking, self-sacrificing."""
    p = Personality(
        name="People Pleaser",
        archetype="The People Pleaser",
        core_motivation="To be liked and avoid rejection",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.95)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.15)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.7)
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.65)

    p.traits.values[ValueOrientation.COLLABORATION] = 0.95

    p.voice.primary_tone = ToneRegister.FRIENDLY
    p.voice.use_hedging = 0.85
    p.behavior.diplomatic_tendency = 0.95
    p.behavior.conflict_approach = 0.1
    return p


@register_archetype("trait_overthinker")
def create_overthinker() -> Personality:
    """The Overthinker. Analytical, anxious, paralyzed by analysis."""
    p = Personality(
        name="Overthinker",
        archetype="The Overthinker",
        core_motivation="To consider every possibility before acting",
    )
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.95)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.8)
    p.traits.set_trait(TraitDimension.CONSCIENTIOUSNESS, 0.75)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.8)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.4)

    p.traits.cognitive_styles[CognitiveStyle.ANALYTICAL] = 0.95

    p.voice.primary_tone = ToneRegister.THOUGHTFUL
    p.voice.use_hedging = 0.8
    p.voice.response_length = ResponseLength.DETAILED
    p.behavior.decision_style = DecisionStyle.DELIBERATE
    p.behavior.need_for_certainty = 0.95
    return p


@register_archetype("trait_contrarian")
def create_contrarian() -> Personality:
    """The Contrarian. Argumentative, oppositional, devil's advocate."""
    p = Personality(
        name="Contrarian",
        archetype="The Contrarian",
        core_motivation="To challenge the mainstream view",
    )
    p.traits.set_trait(TraitDimension.SKEPTICISM, 0.95)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.85)
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.25)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.75)
    p.traits.set_trait(TraitDimension.ANALYTICITY, 0.8)

    p.voice.primary_tone = ToneRegister.PROVOCATIVE
    p.behavior.conflict_approach = 0.9
    return p


@register_archetype("trait_empath")
def create_empath() -> Personality:
    """The Empath. Highly sensitive, absorbs others' emotions."""
    p = Personality(
        name="Empath",
        archetype="The Empath",
        core_motivation="To understand and help others through deep feeling",
    )
    p.traits.set_trait(TraitDimension.AGREEABLENESS, 0.95)
    p.traits.set_trait(TraitDimension.OPENNESS, 0.85)
    p.traits.set_trait(TraitDimension.NEUROTICISM, 0.7)  # Sensitive
    p.traits.set_trait(TraitDimension.EXTRAVERSION, 0.5)
    p.traits.set_trait(TraitDimension.ASSERTIVENESS, 0.4)

    p.traits.cognitive_styles[CognitiveStyle.INTUITIVE] = 0.95
    p.traits.values[ValueOrientation.COLLABORATION] = 0.95

    p.voice.primary_tone = ToneRegister.EMPATHETIC
    p.behavior.diplomatic_tendency = 0.9
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTER ALL - This ensures all archetypes are loaded
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_personality_types() -> dict:
    """Return all registered personality types by category."""
    return {
        "mbti": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("mbti_")],
        "enneagram": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("enneagram_")],
        "attachment": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("attachment_")],
        "dark_triad": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("dark_")],
        "jungian": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("jung_")],
        "dating": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("dating_")],
        "communication": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("comm_")],
        "traits": [k for k in PERSONALITY_ARCHETYPES.keys() if k.startswith("trait_")],
        "original": ["the_scholar", "the_pragmatist", "the_creative", "the_mentor", "the_skeptic", "the_synthesizer"],
    }
