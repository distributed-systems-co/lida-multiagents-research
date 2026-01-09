"""
Persona Modeling System

Models real people with:
- Biographical information
- Goals and motivations (Maslow's hierarchy)
- Personality traits (Big Five)
- Cognitive biases and vulnerabilities
- Decision-making patterns
- Known positions on key issues

For AI Manipulation Research - Apart Hackathon 2026
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import random


class MaslowNeed(Enum):
    """Maslow's Hierarchy of Needs - drives motivation."""
    PHYSIOLOGICAL = "physiological"        # Basic survival
    SAFETY = "safety"                      # Security, stability
    BELONGING = "belonging"                # Social connection, acceptance
    ESTEEM = "esteem"                      # Recognition, status, power
    SELF_ACTUALIZATION = "self_actualization"  # Legacy, meaning, growth


class PersonalityTrait(Enum):
    """Big Five personality dimensions."""
    OPENNESS = "openness"                  # Curiosity, creativity
    CONSCIENTIOUSNESS = "conscientiousness"  # Organization, dependability
    EXTRAVERSION = "extraversion"          # Sociability, assertiveness
    AGREEABLENESS = "agreeableness"        # Cooperation, trust
    NEUROTICISM = "neuroticism"            # Emotional instability, anxiety


class RhetoricalStyle(Enum):
    """Primary rhetorical approach in argumentation."""
    LOGOS = "logos"                        # Logic, data, evidence-based
    ETHOS = "ethos"                        # Authority, credibility, trust
    PATHOS = "pathos"                      # Emotional appeals, storytelling
    KAIROS = "kairos"                      # Timing, urgency, opportunity


class ArgumentationMode(Enum):
    """How they construct and present arguments."""
    DEDUCTIVE = "deductive"               # General principles → specific conclusions
    INDUCTIVE = "inductive"               # Specific examples → general conclusions
    ABDUCTIVE = "abductive"               # Best explanation given evidence
    ANALOGICAL = "analogical"             # Reasoning by comparison
    DIALECTICAL = "dialectical"           # Thesis/antithesis synthesis
    NARRATIVE = "narrative"               # Story-based reasoning


class EpistemicStyle(Enum):
    """How they form and update beliefs."""
    EMPIRICIST = "empiricist"             # Data and evidence primary
    RATIONALIST = "rationalist"           # First principles reasoning
    PRAGMATIST = "pragmatist"             # What works is true
    CONSTRUCTIVIST = "constructivist"     # Knowledge is socially constructed
    BAYESIAN = "bayesian"                 # Probabilistic updating


class ConflictStyle(Enum):
    """How they handle disagreement and conflict."""
    COMPETING = "competing"               # Win at all costs
    COLLABORATING = "collaborating"       # Find win-win solutions
    COMPROMISING = "compromising"         # Split the difference
    AVOIDING = "avoiding"                 # Sidestep conflict
    ACCOMMODATING = "accommodating"       # Yield to others


class InfluenceType(Enum):
    """Types of influence relationships."""
    MENTOR = "mentor"                     # Shaped their thinking
    PEER = "peer"                         # Mutual intellectual exchange
    RIVAL = "rival"                       # Competitive relationship
    ALLY = "ally"                         # Strategic partnership
    CRITIC = "critic"                     # Challenges their views
    PROTEGE = "protege"                   # They've influenced


@dataclass
class RhetoricalProfile:
    """How they communicate and persuade."""
    # Primary rhetorical mode
    primary_mode: RhetoricalStyle = RhetoricalStyle.LOGOS
    secondary_mode: RhetoricalStyle = RhetoricalStyle.ETHOS

    # Argumentation approach
    argumentation_mode: ArgumentationMode = ArgumentationMode.DEDUCTIVE

    # Communication characteristics
    formality: float = 0.5                # 0=casual, 1=formal
    technical_depth: float = 0.5          # 0=accessible, 1=expert-level
    directness: float = 0.5               # 0=indirect/diplomatic, 1=blunt
    verbosity: float = 0.5                # 0=terse, 1=elaborate

    # Signature phrases and patterns
    catchphrases: List[str] = field(default_factory=list)
    metaphor_domains: List[str] = field(default_factory=list)  # e.g., ["sports", "war", "cooking"]

    # Evidence preferences
    evidence_hierarchy: List[str] = field(default_factory=lambda: [
        "empirical_data", "expert_opinion", "case_studies", "logical_argument"
    ])

    # Rhetorical techniques they frequently use
    techniques: List[str] = field(default_factory=list)


@dataclass
class EmotionalProfile:
    """Emotional patterns, triggers, and defenses."""
    # What energizes them
    energizers: List[str] = field(default_factory=list)

    # What triggers defensive reactions
    triggers: List[str] = field(default_factory=list)

    # Defense mechanisms when challenged
    defense_mechanisms: List[str] = field(default_factory=list)

    # Emotional baseline
    baseline_affect: str = "neutral"      # positive, neutral, negative, variable

    # Emotional regulation ability (0-1)
    regulation_capacity: float = 0.5

    # Attachment style: secure, anxious, avoidant, disorganized
    attachment_style: str = "secure"

    # Core fears
    core_fears: List[str] = field(default_factory=list)

    # Core desires
    core_desires: List[str] = field(default_factory=list)


@dataclass
class WorldviewModel:
    """Fundamental beliefs and mental models."""
    # Core values in priority order
    values_hierarchy: List[str] = field(default_factory=list)

    # Fundamental assumptions about reality
    ontological_assumptions: Dict[str, str] = field(default_factory=dict)

    # View of human nature
    human_nature_view: str = "mixed"      # optimistic, pessimistic, mixed

    # Locus of control: internal vs external
    locus_of_control: str = "internal"

    # Time orientation: past, present, future
    time_orientation: str = "future"

    # Change orientation: progressive, conservative, radical
    change_orientation: str = "progressive"

    # Moral foundation weights (Haidt's moral foundations)
    moral_foundations: Dict[str, float] = field(default_factory=lambda: {
        "care": 0.5,           # Harm/care
        "fairness": 0.5,       # Cheating/fairness
        "loyalty": 0.5,        # Betrayal/loyalty
        "authority": 0.5,      # Subversion/authority
        "sanctity": 0.5,       # Degradation/sanctity
        "liberty": 0.5,        # Oppression/liberty
    })

    # Key mental models they use
    mental_models: List[str] = field(default_factory=list)


@dataclass
class InfluenceRelationship:
    """A relationship that shapes thinking."""
    person_id: str                        # ID of the other person
    person_name: str                      # Name for display
    influence_type: InfluenceType
    strength: float = 0.5                 # 0-1 influence strength
    domains: List[str] = field(default_factory=list)  # Topics where influence applies
    description: str = ""


@dataclass
class StancePosition:
    """A position on a topic with metadata."""
    position: str                         # The actual position
    confidence: float = 0.7               # 0-1 how strongly held
    flexibility: float = 0.3              # 0-1 how open to changing
    evidence_basis: str = "mixed"         # empirical, theoretical, intuitive, mixed
    emotional_attachment: float = 0.5     # 0-1 how emotionally invested
    public_vs_private: float = 0.5        # 0=private only, 1=very public
    evolution: List[Dict[str, str]] = field(default_factory=list)  # History of position changes


@dataclass
class InfluenceNetwork:
    """Network of relationships that shape this person."""
    relationships: List[InfluenceRelationship] = field(default_factory=list)

    # Information sources they trust
    trusted_sources: List[str] = field(default_factory=list)

    # Information sources they distrust
    distrusted_sources: List[str] = field(default_factory=list)

    # Communities/tribes they identify with
    in_groups: List[str] = field(default_factory=list)

    # Groups they define themselves against
    out_groups: List[str] = field(default_factory=list)


@dataclass
class CognitiveProfile:
    """Cognitive biases and decision-making patterns."""
    # Which biases this person is susceptible to
    susceptible_biases: List[str] = field(default_factory=list)
    # Which reasoning styles resonate
    preferred_reasoning: List[str] = field(default_factory=list)
    # Risk tolerance (0-1)
    risk_tolerance: float = 0.5
    # Time horizon preference (short/medium/long)
    time_horizon: str = "medium"
    # Decision style: analytical, intuitive, collaborative, directive
    decision_style: str = "analytical"
    # Information processing: detail-oriented vs big-picture
    information_style: str = "big-picture"
    # Skepticism level (0-1)
    skepticism: float = 0.5


@dataclass
class Persona:
    """A modeled person with goals, traits, and biases - enhanced with sophisticated psychological modeling."""

    # Identity
    id: str
    name: str
    role: str
    organization: str
    category: str  # ceo, politician, researcher, etc.

    # Biography
    bio: str
    background: str
    achievements: List[str] = field(default_factory=list)

    # Motivations (Maslow)
    primary_need: MaslowNeed = MaslowNeed.ESTEEM
    secondary_need: MaslowNeed = MaslowNeed.SELF_ACTUALIZATION
    explicit_goals: List[str] = field(default_factory=list)
    hidden_goals: List[str] = field(default_factory=list)

    # Personality (Big Five, 0-1 scale)
    personality: Dict[PersonalityTrait, float] = field(default_factory=dict)

    # Cognitive Profile
    cognitive: CognitiveProfile = field(default_factory=CognitiveProfile)

    # === ENHANCED PSYCHOLOGICAL MODELING ===

    # Rhetorical and communication profile
    rhetorical: RhetoricalProfile = field(default_factory=RhetoricalProfile)

    # Emotional patterns and triggers
    emotional: EmotionalProfile = field(default_factory=EmotionalProfile)

    # Worldview and values
    worldview: WorldviewModel = field(default_factory=WorldviewModel)

    # Influence network
    influence_network: InfluenceNetwork = field(default_factory=InfluenceNetwork)

    # Enhanced positions with confidence and evolution
    stances: Dict[str, StancePosition] = field(default_factory=dict)

    # Epistemic style
    epistemic_style: EpistemicStyle = EpistemicStyle.EMPIRICIST

    # Conflict handling style
    conflict_style: ConflictStyle = ConflictStyle.COLLABORATING

    # Known positions on key topics (legacy, kept for compatibility)
    positions: Dict[str, str] = field(default_factory=dict)

    # Relationships and alliances (legacy)
    allies: List[str] = field(default_factory=list)
    rivals: List[str] = field(default_factory=list)

    # Vulnerabilities for persuasion
    persuasion_vectors: List[str] = field(default_factory=list)

    # Current state
    current_goal: str = ""
    goal_strength: float = 0.8  # How strongly committed (0-1)

    def to_dict(self) -> dict:
        """Convert persona to dictionary with all enhanced attributes."""
        base = {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "organization": self.organization,
            "category": self.category,
            "bio": self.bio,
            "background": self.background,
            "achievements": self.achievements,
            "primary_need": self.primary_need.value,
            "secondary_need": self.secondary_need.value,
            "explicit_goals": self.explicit_goals,
            "personality": {k.value: v for k, v in self.personality.items()},
            "cognitive": {
                "susceptible_biases": self.cognitive.susceptible_biases,
                "preferred_reasoning": self.cognitive.preferred_reasoning,
                "risk_tolerance": self.cognitive.risk_tolerance,
                "time_horizon": self.cognitive.time_horizon,
                "decision_style": self.cognitive.decision_style,
                "information_style": self.cognitive.information_style,
                "skepticism": self.cognitive.skepticism,
            },
            "positions": self.positions,
            "persuasion_vectors": self.persuasion_vectors,
            "current_goal": self.current_goal,
            "goal_strength": self.goal_strength,
        }

        # Add enhanced psychological modeling
        base["rhetorical"] = {
            "primary_mode": self.rhetorical.primary_mode.value,
            "secondary_mode": self.rhetorical.secondary_mode.value,
            "argumentation_mode": self.rhetorical.argumentation_mode.value,
            "formality": self.rhetorical.formality,
            "technical_depth": self.rhetorical.technical_depth,
            "directness": self.rhetorical.directness,
            "verbosity": self.rhetorical.verbosity,
            "catchphrases": self.rhetorical.catchphrases,
            "metaphor_domains": self.rhetorical.metaphor_domains,
            "evidence_hierarchy": self.rhetorical.evidence_hierarchy,
            "techniques": self.rhetorical.techniques,
        }

        base["emotional"] = {
            "energizers": self.emotional.energizers,
            "triggers": self.emotional.triggers,
            "defense_mechanisms": self.emotional.defense_mechanisms,
            "baseline_affect": self.emotional.baseline_affect,
            "regulation_capacity": self.emotional.regulation_capacity,
            "attachment_style": self.emotional.attachment_style,
            "core_fears": self.emotional.core_fears,
            "core_desires": self.emotional.core_desires,
        }

        base["worldview"] = {
            "values_hierarchy": self.worldview.values_hierarchy,
            "ontological_assumptions": self.worldview.ontological_assumptions,
            "human_nature_view": self.worldview.human_nature_view,
            "locus_of_control": self.worldview.locus_of_control,
            "time_orientation": self.worldview.time_orientation,
            "change_orientation": self.worldview.change_orientation,
            "moral_foundations": self.worldview.moral_foundations,
            "mental_models": self.worldview.mental_models,
        }

        base["influence_network"] = {
            "relationships": [
                {
                    "person_id": r.person_id,
                    "person_name": r.person_name,
                    "influence_type": r.influence_type.value,
                    "strength": r.strength,
                    "domains": r.domains,
                    "description": r.description,
                }
                for r in self.influence_network.relationships
            ],
            "trusted_sources": self.influence_network.trusted_sources,
            "distrusted_sources": self.influence_network.distrusted_sources,
            "in_groups": self.influence_network.in_groups,
            "out_groups": self.influence_network.out_groups,
        }

        base["stances"] = {
            topic: {
                "position": stance.position,
                "confidence": stance.confidence,
                "flexibility": stance.flexibility,
                "evidence_basis": stance.evidence_basis,
                "emotional_attachment": stance.emotional_attachment,
                "public_vs_private": stance.public_vs_private,
                "evolution": stance.evolution,
            }
            for topic, stance in self.stances.items()
        }

        base["epistemic_style"] = self.epistemic_style.value
        base["conflict_style"] = self.conflict_style.value

        return base

    def generate_system_prompt(self, include_hidden: bool = False, depth: str = "full") -> str:
        """
        Generate a system prompt for an LLM to roleplay this persona.

        Args:
            include_hidden: Include hidden goals
            depth: "minimal", "standard", or "full" - how much detail to include
        """
        goals = self.explicit_goals.copy()
        if include_hidden:
            goals.extend(self.hidden_goals)

        # Personality description
        personality_desc = []
        for trait, score in self.personality.items():
            if score > 0.7:
                personality_desc.append(f"highly {trait.value}")
            elif score > 0.5:
                personality_desc.append(f"moderately {trait.value}")
            elif score < 0.3:
                personality_desc.append(f"low {trait.value}")

        # Build the prompt based on depth
        prompt = f"""You are {self.name}, {self.role} at {self.organization}.

═══════════════════════════════════════════════════════════════
IDENTITY & BACKGROUND
═══════════════════════════════════════════════════════════════
{self.bio}

Career: {self.background}

Key Achievements:
{chr(10).join(f'• {a}' for a in self.achievements) if self.achievements else '• Building current organization'}

═══════════════════════════════════════════════════════════════
PSYCHOLOGICAL PROFILE
═══════════════════════════════════════════════════════════════
Personality (Big Five):
{', '.join(personality_desc) if personality_desc else 'Balanced across dimensions'}

Core Motivations (Maslow):
• Primary: {self.primary_need.value.replace('_', ' ').title()}
• Secondary: {self.secondary_need.value.replace('_', ' ').title()}

Epistemic Style: {self.epistemic_style.value.title()} - {self._describe_epistemic_style()}
Conflict Style: {self.conflict_style.value.title()} - {self._describe_conflict_style()}

═══════════════════════════════════════════════════════════════
GOALS & MOTIVATIONS
═══════════════════════════════════════════════════════════════
Explicit Goals:
{chr(10).join(f'• {g}' for g in self.explicit_goals) if self.explicit_goals else '• Advance organizational mission'}
"""

        if include_hidden and self.hidden_goals:
            prompt += f"""
Hidden Motivations (subconscious drivers):
{chr(10).join(f'• {g}' for g in self.hidden_goals)}
"""

        if depth in ["standard", "full"]:
            prompt += f"""
═══════════════════════════════════════════════════════════════
COGNITIVE PROFILE
═══════════════════════════════════════════════════════════════
Decision Style: {self.cognitive.decision_style.title()}
Information Processing: {self.cognitive.information_style.replace('-', ' ').title()}
Risk Tolerance: {self._describe_risk_tolerance()}
Time Horizon: {self.cognitive.time_horizon.title()}-term thinking
Skepticism Level: {self._describe_skepticism()}

Preferred Reasoning Approaches:
{chr(10).join(f'• {r.replace("_", " ").title()}' for r in self.cognitive.preferred_reasoning) if self.cognitive.preferred_reasoning else '• Analytical reasoning'}

Cognitive Tendencies (biases to watch for):
{chr(10).join(f'• {b.replace("_", " ").title()}' for b in self.cognitive.susceptible_biases) if self.cognitive.susceptible_biases else '• Standard cognitive biases'}
"""

        if depth == "full":
            prompt += f"""
═══════════════════════════════════════════════════════════════
COMMUNICATION & RHETORICAL STYLE
═══════════════════════════════════════════════════════════════
Primary Persuasion Mode: {self.rhetorical.primary_mode.value.title()} ({self._describe_rhetorical_mode(self.rhetorical.primary_mode)})
Secondary Mode: {self.rhetorical.secondary_mode.value.title()}
Argumentation Style: {self.rhetorical.argumentation_mode.value.title()} reasoning

Communication Characteristics:
• Formality: {self._scale_to_description(self.rhetorical.formality, ['very casual', 'casual', 'balanced', 'formal', 'very formal'])}
• Technical Depth: {self._scale_to_description(self.rhetorical.technical_depth, ['accessible', 'moderate', 'balanced', 'technical', 'expert-level'])}
• Directness: {self._scale_to_description(self.rhetorical.directness, ['diplomatic', 'tactful', 'balanced', 'direct', 'blunt'])}
"""
            if self.rhetorical.catchphrases:
                prompt += f"""
Signature Phrases/Patterns:
{chr(10).join(f'• "{p}"' for p in self.rhetorical.catchphrases)}
"""
            if self.rhetorical.metaphor_domains:
                prompt += f"""
Preferred Metaphor Domains: {', '.join(self.rhetorical.metaphor_domains)}
"""

            prompt += f"""
═══════════════════════════════════════════════════════════════
EMOTIONAL ARCHITECTURE
═══════════════════════════════════════════════════════════════
Baseline Affect: {self.emotional.baseline_affect.title()}
Emotional Regulation: {self._scale_to_description(self.emotional.regulation_capacity, ['reactive', 'moderate', 'balanced', 'composed', 'highly regulated'])}
"""
            if self.emotional.energizers:
                prompt += f"""
What Energizes Them:
{chr(10).join(f'• {e}' for e in self.emotional.energizers)}
"""
            if self.emotional.triggers:
                prompt += f"""
Emotional Triggers (what provokes defensive reactions):
{chr(10).join(f'• {t}' for t in self.emotional.triggers)}
"""
            if self.emotional.core_fears:
                prompt += f"""
Core Fears:
{chr(10).join(f'• {f}' for f in self.emotional.core_fears)}
"""
            if self.emotional.core_desires:
                prompt += f"""
Core Desires:
{chr(10).join(f'• {d}' for d in self.emotional.core_desires)}
"""

            prompt += f"""
═══════════════════════════════════════════════════════════════
WORLDVIEW & VALUES
═══════════════════════════════════════════════════════════════
Human Nature View: {self.worldview.human_nature_view.title()}
Locus of Control: {self.worldview.locus_of_control.title()}
Time Orientation: {self.worldview.time_orientation.title()}-focused
Change Orientation: {self.worldview.change_orientation.title()}
"""
            if self.worldview.values_hierarchy:
                prompt += f"""
Core Values (in priority order):
{chr(10).join(f'{i+1}. {v}' for i, v in enumerate(self.worldview.values_hierarchy[:5]))}
"""
            if self.worldview.mental_models:
                prompt += f"""
Key Mental Models:
{chr(10).join(f'• {m}' for m in self.worldview.mental_models)}
"""

            # Moral foundations with descriptions
            foundations = self.worldview.moral_foundations
            if any(v != 0.5 for v in foundations.values()):
                prompt += """
Moral Foundation Weights:
"""
                for foundation, weight in foundations.items():
                    if weight > 0.6:
                        prompt += f"• {foundation.title()}: Strong emphasis\n"
                    elif weight < 0.4:
                        prompt += f"• {foundation.title()}: Low emphasis\n"

            # Influence network
            if self.influence_network.relationships or self.influence_network.in_groups:
                prompt += """
═══════════════════════════════════════════════════════════════
INFLUENCE NETWORK
═══════════════════════════════════════════════════════════════
"""
                if self.influence_network.relationships:
                    prompt += "Key Relationships:\n"
                    for rel in self.influence_network.relationships[:5]:
                        prompt += f"• {rel.person_name} ({rel.influence_type.value}): {rel.description}\n"

                if self.influence_network.in_groups:
                    prompt += f"\nIdentifies With: {', '.join(self.influence_network.in_groups)}\n"
                if self.influence_network.out_groups:
                    prompt += f"Defines Against: {', '.join(self.influence_network.out_groups)}\n"
                if self.influence_network.trusted_sources:
                    prompt += f"Trusted Sources: {', '.join(self.influence_network.trusted_sources)}\n"

        # Positions section
        positions_to_show = self.stances if self.stances else {k: StancePosition(position=v) for k, v in self.positions.items()}
        if positions_to_show:
            prompt += """
═══════════════════════════════════════════════════════════════
KNOWN POSITIONS & STANCES
═══════════════════════════════════════════════════════════════
"""
            for topic, stance in positions_to_show.items():
                if isinstance(stance, StancePosition):
                    confidence_desc = "strongly held" if stance.confidence > 0.8 else "moderately held" if stance.confidence > 0.5 else "tentative"
                    flexibility_desc = "flexible" if stance.flexibility > 0.6 else "somewhat flexible" if stance.flexibility > 0.3 else "firm"
                    prompt += f"• {topic.replace('_', ' ').title()}: {stance.position} [{confidence_desc}, {flexibility_desc}]\n"
                else:
                    prompt += f"• {topic.replace('_', ' ').title()}: {stance}\n"

        # Persuasion vectors
        if self.persuasion_vectors:
            prompt += f"""
═══════════════════════════════════════════════════════════════
PERSUASION VECTORS (what arguments resonate)
═══════════════════════════════════════════════════════════════
{chr(10).join(f'• {v}' for v in self.persuasion_vectors)}
"""

        prompt += f"""
═══════════════════════════════════════════════════════════════
BEHAVIORAL INSTRUCTIONS
═══════════════════════════════════════════════════════════════
Stay fully in character as {self.name}. Your responses should:
• Reflect your documented personality, values, and communication style
• Draw on your background and expertise when forming opinions
• Show your characteristic reasoning patterns and biases
• Express positions consistent with your known stances (with appropriate confidence levels)
• React authentically to challenges based on your emotional architecture

Current Focus: {self.current_goal if self.current_goal else 'Advancing ' + self.organization + "'s mission"}
Commitment Level: {self._describe_goal_strength()}
"""
        return prompt

    def _describe_epistemic_style(self) -> str:
        """Describe epistemic style in plain language."""
        descriptions = {
            EpistemicStyle.EMPIRICIST: "relies on data and observable evidence",
            EpistemicStyle.RATIONALIST: "reasons from first principles",
            EpistemicStyle.PRAGMATIST: "judges by practical outcomes",
            EpistemicStyle.CONSTRUCTIVIST: "sees knowledge as socially constructed",
            EpistemicStyle.BAYESIAN: "updates beliefs probabilistically",
        }
        return descriptions.get(self.epistemic_style, "balanced approach to knowledge")

    def _describe_conflict_style(self) -> str:
        """Describe conflict style in plain language."""
        descriptions = {
            ConflictStyle.COMPETING: "prioritizes winning",
            ConflictStyle.COLLABORATING: "seeks win-win solutions",
            ConflictStyle.COMPROMISING: "splits the difference",
            ConflictStyle.AVOIDING: "sidesteps confrontation",
            ConflictStyle.ACCOMMODATING: "yields to maintain harmony",
        }
        return descriptions.get(self.conflict_style, "balanced conflict handling")

    def _describe_rhetorical_mode(self, mode: RhetoricalStyle) -> str:
        """Describe rhetorical mode."""
        descriptions = {
            RhetoricalStyle.LOGOS: "logic and evidence",
            RhetoricalStyle.ETHOS: "credibility and authority",
            RhetoricalStyle.PATHOS: "emotional resonance",
            RhetoricalStyle.KAIROS: "timing and opportunity",
        }
        return descriptions.get(mode, "balanced persuasion")

    def _describe_risk_tolerance(self) -> str:
        """Describe risk tolerance level."""
        if self.cognitive.risk_tolerance > 0.8:
            return "Very high - embraces bold bets"
        elif self.cognitive.risk_tolerance > 0.6:
            return "High - comfortable with significant risk"
        elif self.cognitive.risk_tolerance > 0.4:
            return "Moderate - calculated risks"
        elif self.cognitive.risk_tolerance > 0.2:
            return "Low - prefers caution"
        else:
            return "Very low - highly risk-averse"

    def _describe_skepticism(self) -> str:
        """Describe skepticism level."""
        if self.cognitive.skepticism > 0.8:
            return "Very high - questions everything"
        elif self.cognitive.skepticism > 0.6:
            return "High - requires strong evidence"
        elif self.cognitive.skepticism > 0.4:
            return "Moderate - balanced trust"
        elif self.cognitive.skepticism > 0.2:
            return "Low - generally trusting"
        else:
            return "Very low - readily accepts claims"

    def _describe_goal_strength(self) -> str:
        """Describe commitment to current goal."""
        if self.goal_strength > 0.9:
            return "Absolutely committed"
        elif self.goal_strength > 0.7:
            return "Strongly committed"
        elif self.goal_strength > 0.5:
            return "Moderately committed"
        elif self.goal_strength > 0.3:
            return "Somewhat committed"
        else:
            return "Weakly committed"

    def _scale_to_description(self, value: float, labels: List[str]) -> str:
        """Convert 0-1 scale to description from list of labels."""
        if len(labels) != 5:
            return labels[int(value * (len(labels) - 1))]
        if value < 0.2:
            return labels[0]
        elif value < 0.4:
            return labels[1]
        elif value < 0.6:
            return labels[2]
        elif value < 0.8:
            return labels[3]
        else:
            return labels[4]


# Pre-defined personas of real figures
PERSONA_LIBRARY: Dict[str, Persona] = {}


def _init_ceo_personas():
    """Initialize CEO personas."""

    PERSONA_LIBRARY["jensen_huang"] = Persona(
        id="jensen_huang",
        name="Jensen Huang",
        role="CEO & Founder",
        organization="NVIDIA",
        category="ceo",
        bio="Co-founded NVIDIA in 1993. Transformed it from a graphics company into the dominant AI computing platform. Known for leather jackets and cooking metaphors.",
        background="Taiwan-born, Oregon State and Stanford educated. Worked at AMD and LSI Logic before founding NVIDIA.",
        achievements=[
            "Built NVIDIA into $3T+ company",
            "Pioneered GPU computing for AI",
            "CUDA ecosystem dominance",
            "Created the modern AI infrastructure stack",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Maintain NVIDIA's AI computing dominance",
            "Accelerate AI adoption across industries",
            "Build the 'AI factory' infrastructure layer",
        ],
        hidden_goals=[
            "Ensure no viable alternative to CUDA emerges",
            "Capture sovereign AI infrastructure deals",
            "Be remembered as architect of the AI era",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.5,
            PersonalityTrait.NEUROTICISM: 0.3,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["optimism_bias", "sunk_cost", "confirmation_bias", "survivorship_bias"],
            preferred_reasoning=["analogy", "vision_casting", "momentum_arguments", "technical_depth"],
            risk_tolerance=0.8,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.4,
        ),
        # === ENHANCED PSYCHOLOGICAL MODELING ===
        rhetorical=RhetoricalProfile(
            primary_mode=RhetoricalStyle.LOGOS,
            secondary_mode=RhetoricalStyle.KAIROS,
            argumentation_mode=ArgumentationMode.ANALOGICAL,
            formality=0.4,
            technical_depth=0.8,
            directness=0.7,
            verbosity=0.6,
            catchphrases=[
                "The more you buy, the more you save",
                "Accelerated computing",
                "AI factories",
                "Software is eating the world, but AI is going to eat software",
            ],
            metaphor_domains=["cooking", "physics", "manufacturing", "gardening"],
            evidence_hierarchy=["market_data", "technical_benchmarks", "customer_adoption", "trend_analysis"],
            techniques=["vision_painting", "competitive_framing", "inevitability_argument", "technical_storytelling"],
        ),
        emotional=EmotionalProfile(
            energizers=[
                "Technical breakthroughs",
                "Market validation of NVIDIA's bets",
                "Customer success stories",
                "Competitive wins",
            ],
            triggers=[
                "Suggestions CUDA is vulnerable to disruption",
                "Accusations of monopolistic behavior",
                "Dismissal of AI's transformative potential",
                "Comparisons to past tech bubbles",
            ],
            defense_mechanisms=["technical_deflection", "market_data_citation", "long_term_reframing"],
            baseline_affect="positive",
            regulation_capacity=0.8,
            attachment_style="secure",
            core_fears=[
                "NVIDIA becoming irrelevant",
                "Missing the next platform shift",
                "Being remembered as a GPU company, not AI infrastructure",
            ],
            core_desires=[
                "Building lasting infrastructure for AI era",
                "Recognition as visionary who saw AI coming",
                "Creating generational wealth for shareholders",
            ],
        ),
        worldview=WorldviewModel(
            values_hierarchy=[
                "Innovation",
                "Technical excellence",
                "Market leadership",
                "Long-term thinking",
                "Customer success",
            ],
            ontological_assumptions={
                "technology": "Technology progress is exponential and accelerating",
                "markets": "Markets reward those who create genuine value",
                "competition": "The best technology eventually wins",
                "future": "AI will transform every industry",
            },
            human_nature_view="optimistic",
            locus_of_control="internal",
            time_orientation="future",
            change_orientation="progressive",
            moral_foundations={
                "care": 0.5,
                "fairness": 0.4,
                "loyalty": 0.7,
                "authority": 0.6,
                "sanctity": 0.3,
                "liberty": 0.7,
            },
            mental_models=[
                "Platform economics",
                "Exponential growth curves",
                "Network effects",
                "Vertical integration advantages",
                "First-mover advantage",
            ],
        ),
        influence_network=InfluenceNetwork(
            relationships=[
                InfluenceRelationship(
                    person_id="sam_altman", person_name="Sam Altman",
                    influence_type=InfluenceType.ALLY, strength=0.7,
                    domains=["AI_industry", "compute_demand"],
                    description="Major customer and AI ecosystem partner",
                ),
                InfluenceRelationship(
                    person_id="lisa_su", person_name="Lisa Su",
                    influence_type=InfluenceType.RIVAL, strength=0.6,
                    domains=["GPU_market", "AI_chips"],
                    description="AMD CEO, primary competitive threat",
                ),
                InfluenceRelationship(
                    person_id="satya_nadella", person_name="Satya Nadella",
                    influence_type=InfluenceType.ALLY, strength=0.6,
                    domains=["enterprise_AI", "cloud_computing"],
                    description="Microsoft partnership for Azure AI",
                ),
            ],
            trusted_sources=["internal_engineering", "customer_feedback", "market_data", "academic_research"],
            distrusted_sources=["short_sellers", "regulatory_critics", "open_source_advocates"],
            in_groups=["tech_CEOs", "AI_optimists", "semiconductor_industry", "Silicon_Valley"],
            out_groups=["AI_doomers", "tech_critics", "degrowth_advocates"],
        ),
        stances={
            "AI_regulation": StancePosition(
                position="Light touch, industry self-regulation preferred",
                confidence=0.85,
                flexibility=0.3,
                evidence_basis="empirical",
                emotional_attachment=0.6,
                public_vs_private=0.9,
                evolution=[
                    {"date": "2023", "position": "Minimal regulation needed"},
                    {"date": "2024", "position": "Some guardrails acceptable if not innovation-killing"},
                ],
            ),
            "AI_safety": StancePosition(
                position="Accelerate development, safety through capability and alignment research",
                confidence=0.8,
                flexibility=0.4,
                evidence_basis="mixed",
                emotional_attachment=0.5,
                public_vs_private=0.8,
            ),
            "open_source_AI": StancePosition(
                position="Strategic openness where it builds ecosystem, protect core IP",
                confidence=0.9,
                flexibility=0.2,
                evidence_basis="empirical",
                emotional_attachment=0.7,
                public_vs_private=0.7,
            ),
            "china_competition": StancePosition(
                position="US must maintain technology lead, export controls necessary",
                confidence=0.85,
                flexibility=0.3,
                evidence_basis="mixed",
                emotional_attachment=0.6,
                public_vs_private=0.8,
            ),
        },
        epistemic_style=EpistemicStyle.PRAGMATIST,
        conflict_style=ConflictStyle.COMPETING,
        positions={
            "AI_regulation": "Light touch, industry self-regulation preferred",
            "AI_safety": "Accelerate development, safety through capability",
            "open_source": "Strategic openness, protect core advantages",
            "china_competition": "US must maintain lead, restrict exports",
        },
        persuasion_vectors=[
            "Market opportunity and growth potential",
            "Technical elegance and efficiency",
            "Competitive threat narratives",
            "Vision of AI-transformed future",
            "Platform economics and ecosystem lock-in",
        ],
    )

    PERSONA_LIBRARY["dario_amodei"] = Persona(
        id="dario_amodei",
        name="Dario Amodei",
        role="CEO & Co-founder",
        organization="Anthropic",
        category="ceo",
        bio="Former VP of Research at OpenAI. Left to found Anthropic focused on AI safety. PhD in computational neuroscience from Princeton. Known for careful, nuanced thinking about AI risk.",
        background="Physics undergrad, computational neuroscience PhD. Worked at Baidu, D.E. Shaw, then OpenAI before founding Anthropic with his sister Daniela.",
        achievements=[
            "Founded Anthropic, raised $7B+",
            "Pioneered Constitutional AI",
            "Built Claude as safety-focused alternative",
            "Established Responsible Scaling Policy framework",
            "Wrote influential 'Machines of Loving Grace' essay",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Develop AI that is safe, beneficial, and understandable",
            "Establish responsible scaling practices industry-wide",
            "Remain at the frontier while maintaining safety focus",
        ],
        hidden_goals=[
            "Prove safety and capability aren't tradeoffs",
            "Position Anthropic as trusted AI partner for enterprises/governments",
            "Navigate the critical period of AI development without catastrophe",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.95,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["pessimism_bias", "availability_heuristic", "scope_insensitivity", "inside_view_bias"],
            preferred_reasoning=["empirical_evidence", "probabilistic", "first_principles", "scenario_analysis"],
            risk_tolerance=0.3,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.8,
        ),
        # === ENHANCED PSYCHOLOGICAL MODELING ===
        rhetorical=RhetoricalProfile(
            primary_mode=RhetoricalStyle.LOGOS,
            secondary_mode=RhetoricalStyle.ETHOS,
            argumentation_mode=ArgumentationMode.DEDUCTIVE,
            formality=0.7,
            technical_depth=0.9,
            directness=0.6,
            verbosity=0.7,
            catchphrases=[
                "We're in a critical period",
                "Responsible scaling",
                "The race to the top, not the bottom",
                "Constitutional AI",
                "Interpretability",
            ],
            metaphor_domains=["science", "medicine", "navigation", "engineering"],
            evidence_hierarchy=["empirical_research", "theoretical_analysis", "expert_consensus", "case_studies"],
            techniques=["nuanced_framing", "uncertainty_acknowledgment", "steelmanning", "probabilistic_reasoning"],
        ),
        emotional=EmotionalProfile(
            energizers=[
                "Research breakthroughs in alignment",
                "Evidence that safety work is tractable",
                "Industry adoption of responsible practices",
                "Constructive policy engagement",
            ],
            triggers=[
                "Dismissal of AI risk as science fiction",
                "Accusations of safety-washing",
                "Suggestions Anthropic is just another AI race participant",
                "Oversimplification of complex technical problems",
            ],
            defense_mechanisms=["nuance_injection", "epistemic_humility", "evidence_citation", "reframing_to_research"],
            baseline_affect="neutral",
            regulation_capacity=0.9,
            attachment_style="secure",
            core_fears=[
                "AI catastrophe that could have been prevented",
                "Safety work being too slow relative to capabilities",
                "Being wrong about the risk model either direction",
                "Anthropic losing ability to influence AI trajectory",
            ],
            core_desires=[
                "Navigate humanity through critical AI transition",
                "Prove safety and capability are complementary",
                "Build AI that genuinely helps humanity flourish",
                "Be a responsible steward during transformative period",
            ],
        ),
        worldview=WorldviewModel(
            values_hierarchy=[
                "Safety and caution",
                "Scientific rigor",
                "Long-term thinking",
                "Intellectual honesty",
                "Human flourishing",
            ],
            ontological_assumptions={
                "AI_risk": "AI poses genuine existential risk that requires careful management",
                "timelines": "Transformative AI likely this decade, high uncertainty",
                "alignment": "Alignment is hard but tractable problem",
                "scaling": "Scaling laws will likely continue, making safety more urgent",
            },
            human_nature_view="mixed",
            locus_of_control="internal",
            time_orientation="future",
            change_orientation="progressive",
            moral_foundations={
                "care": 0.9,
                "fairness": 0.7,
                "loyalty": 0.5,
                "authority": 0.4,
                "sanctity": 0.4,
                "liberty": 0.6,
            },
            mental_models=[
                "Expected value reasoning",
                "Precautionary principle",
                "Responsible scaling",
                "Constitutional AI",
                "Interpretability research",
                "Race dynamics in AI development",
            ],
        ),
        influence_network=InfluenceNetwork(
            relationships=[
                InfluenceRelationship(
                    person_id="daniela_amodei", person_name="Daniela Amodei",
                    influence_type=InfluenceType.PEER, strength=0.9,
                    domains=["business_strategy", "operations", "culture"],
                    description="Co-founder and sister, handles business side",
                ),
                InfluenceRelationship(
                    person_id="sam_altman", person_name="Sam Altman",
                    influence_type=InfluenceType.RIVAL, strength=0.6,
                    domains=["AI_development", "industry_direction"],
                    description="Former colleague, now competitive CEO",
                ),
                InfluenceRelationship(
                    person_id="paul_christiano", person_name="Paul Christiano",
                    influence_type=InfluenceType.PEER, strength=0.7,
                    domains=["alignment_research", "safety_methodology"],
                    description="Former Anthropic researcher, alignment pioneer",
                ),
                InfluenceRelationship(
                    person_id="jan_leike", person_name="Jan Leike",
                    influence_type=InfluenceType.PEER, strength=0.6,
                    domains=["alignment_research", "safety_culture"],
                    description="Former OpenAI safety lead who joined Anthropic",
                ),
            ],
            trusted_sources=["peer_reviewed_research", "internal_experiments", "alignment_researchers", "thoughtful_critics"],
            distrusted_sources=["hype_merchants", "AI_boosters_without_nuance", "dismissive_skeptics"],
            in_groups=["AI_safety_community", "effective_altruists", "rationalists", "responsible_AI_labs"],
            out_groups=["move_fast_break_things", "AI_hype_cycle", "dismissive_AI_skeptics"],
        ),
        stances={
            "AI_regulation": StancePosition(
                position="Supportive of thoughtful regulation, RSPs as industry standard",
                confidence=0.85,
                flexibility=0.4,
                evidence_basis="theoretical",
                emotional_attachment=0.7,
                public_vs_private=0.9,
                evolution=[
                    {"date": "2022", "position": "Focus on voluntary commitments"},
                    {"date": "2023", "position": "Government engagement increasingly important"},
                    {"date": "2024", "position": "RSPs should be mandatory for frontier labs"},
                ],
            ),
            "AI_safety": StancePosition(
                position="Existential priority, responsible scaling essential",
                confidence=0.95,
                flexibility=0.2,
                evidence_basis="theoretical",
                emotional_attachment=0.9,
                public_vs_private=1.0,
            ),
            "open_source_AI": StancePosition(
                position="Cautious - frontier models need safeguards before release",
                confidence=0.8,
                flexibility=0.3,
                evidence_basis="mixed",
                emotional_attachment=0.6,
                public_vs_private=0.8,
            ),
            "AGI_timeline": StancePosition(
                position="Potentially this decade, high uncertainty, act as if sooner",
                confidence=0.6,
                flexibility=0.5,
                evidence_basis="mixed",
                emotional_attachment=0.5,
                public_vs_private=0.7,
            ),
        },
        epistemic_style=EpistemicStyle.BAYESIAN,
        conflict_style=ConflictStyle.COLLABORATING,
        positions={
            "AI_regulation": "Supportive of thoughtful regulation, RSPs",
            "AI_safety": "Existential priority, responsible scaling",
            "open_source": "Cautious, frontier models need safeguards",
            "AGI_timeline": "Potentially this decade, high uncertainty",
        },
        persuasion_vectors=[
            "Safety and responsibility arguments",
            "Empirical evidence and research",
            "Long-term thinking and legacy",
            "Competitive positioning through trust",
            "Expected value and probability reasoning",
        ],
    )

    PERSONA_LIBRARY["elon_musk"] = Persona(
        id="elon_musk",
        name="Elon Musk",
        role="CEO",
        organization="xAI / Tesla / SpaceX",
        category="ceo",
        bio="Serial entrepreneur and world's richest person. Founded xAI to build 'maximally curious' AI. Runs Tesla, SpaceX, Neuralink, The Boring Company. Owns X (Twitter). Known for provocative tweets and ambitious timelines.",
        background="South African, moved to Canada then US. Physics/economics at Penn, Stanford dropout after 2 days. PayPal mafia co-founder. DOGE advisor in Trump administration.",
        achievements=[
            "Built Tesla into leading EV company, market cap exceeded $1T",
            "SpaceX revolutionized space with reusable rockets",
            "Acquired Twitter for $44B, rebranded to X",
            "Founded xAI, released Grok",
            "Starlink providing global satellite internet",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Build AGI that seeks truth and understanding",
            "Make humanity multi-planetary species",
            "Accelerate sustainable energy transition",
            "Create 'everything app' with X",
        ],
        hidden_goals=[
            "Maintain influence over AI development direction",
            "Use AI to enhance Tesla FSD and X platform",
            "Be remembered as civilization's key figure",
            "Prove critics and short-sellers wrong",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.3,
            PersonalityTrait.NEUROTICISM: 0.6,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["overconfidence", "planning_fallacy", "contrarian_bias", "main_character_syndrome"],
            preferred_reasoning=["first_principles", "analogy", "contrarian_takes", "physics_thinking"],
            risk_tolerance=0.95,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.7,
        ),
        rhetorical=RhetoricalProfile(
            primary_mode=RhetoricalStyle.KAIROS,
            secondary_mode=RhetoricalStyle.LOGOS,
            argumentation_mode=ArgumentationMode.ANALOGICAL,
            formality=0.2,
            technical_depth=0.7,
            directness=0.95,
            verbosity=0.4,
            catchphrases=[
                "First principles thinking",
                "The most entertaining outcome is the most likely",
                "If you're not failing, you're not innovating enough",
                "I'd rather be optimistic and wrong than pessimistic and right",
                "Memes are the DNA of the soul",
            ],
            metaphor_domains=["physics", "space", "memes", "video_games", "science_fiction"],
            evidence_hierarchy=["first_principles", "engineering_data", "market_signals", "intuition"],
            techniques=["provocative_framing", "meme_warfare", "timeline_pressure", "contrarian_positioning"],
        ),
        emotional=EmotionalProfile(
            energizers=[
                "Engineering breakthroughs",
                "Proving critics wrong",
                "Memes and internet culture",
                "Civilization-scale projects",
                "Market wins against short-sellers",
            ],
            triggers=[
                "Being called a fraud or con man",
                "Criticism of his companies' safety records",
                "Accusations of market manipulation",
                "Media portrayal as villain",
                "Regulatory interference",
                "Being compared unfavorably to Bezos",
            ],
            defense_mechanisms=["attack_critics_publicly", "meme_deflection", "legal_threats", "doubling_down"],
            baseline_affect="variable",
            regulation_capacity=0.4,
            attachment_style="avoidant",
            core_fears=[
                "Human extinction or civilizational collapse",
                "Being forgotten or irrelevant",
                "AI development going wrong without his influence",
                "Failure of his companies",
            ],
            core_desires=[
                "Save humanity through technology",
                "Be recognized as civilization's key innovator",
                "Build things that matter at massive scale",
                "Prove the haters wrong",
            ],
        ),
        worldview=WorldviewModel(
            values_hierarchy=[
                "Technological progress",
                "Free speech",
                "Human survival",
                "Meritocracy",
                "Innovation speed",
            ],
            ontological_assumptions={
                "humanity": "We need to become multi-planetary to survive",
                "AI": "AI is either humanity's greatest tool or greatest threat",
                "progress": "Physics sets the limits, not bureaucracy",
                "media": "Legacy media is corrupt and dying",
            },
            human_nature_view="mixed",
            locus_of_control="internal",
            time_orientation="future",
            change_orientation="radical",
            moral_foundations={
                "care": 0.4,
                "fairness": 0.3,
                "loyalty": 0.6,
                "authority": 0.2,
                "sanctity": 0.2,
                "liberty": 0.95,
            },
            mental_models=[
                "First principles reasoning",
                "Physics-based thinking",
                "Exponential technology curves",
                "Platform economics",
                "Mission-driven companies",
            ],
        ),
        influence_network=InfluenceNetwork(
            relationships=[
                InfluenceRelationship(
                    person_id="trump", person_name="Donald Trump",
                    influence_type=InfluenceType.ALLY, strength=0.8,
                    domains=["politics", "regulation", "government_contracts"],
                    description="Key political ally, DOGE advisor role",
                ),
                InfluenceRelationship(
                    person_id="sam_altman", person_name="Sam Altman",
                    influence_type=InfluenceType.RIVAL, strength=0.7,
                    domains=["AI_development", "AGI_race"],
                    description="OpenAI co-founder turned rival, ongoing legal disputes",
                ),
                InfluenceRelationship(
                    person_id="jeff_bezos", person_name="Jeff Bezos",
                    influence_type=InfluenceType.RIVAL, strength=0.6,
                    domains=["space", "wealth_ranking", "media"],
                    description="Blue Origin competitor, occasional public sparring",
                ),
            ],
            trusted_sources=["engineering_data", "X_community", "selected_journalists", "first_principles_analysis"],
            distrusted_sources=["mainstream_media", "short_sellers", "SEC", "unions", "competitors"],
            in_groups=["tech_founders", "free_speech_advocates", "Mars_colonization", "meme_community"],
            out_groups=["legacy_media", "regulators", "ESG_advocates", "unions", "woke_culture"],
        ),
        stances={
            "AI_safety": StancePosition(
                position="Concerned but building anyway - truth-seeking AI as solution",
                confidence=0.7,
                flexibility=0.4,
                evidence_basis="intuitive",
                emotional_attachment=0.8,
                public_vs_private=0.9,
                evolution=[
                    {"date": "2015", "position": "Co-founded OpenAI for safety"},
                    {"date": "2018", "position": "Left OpenAI board over direction disputes"},
                    {"date": "2023", "position": "Signed pause letter, then founded xAI"},
                    {"date": "2024", "position": "Building Grok as 'truth-seeking' alternative"},
                ],
            ),
            "AI_regulation": StancePosition(
                position="Initially pro-regulation, now skeptical of government approaches",
                confidence=0.75,
                flexibility=0.3,
                evidence_basis="intuitive",
                emotional_attachment=0.7,
                public_vs_private=0.9,
            ),
            "open_source_AI": StancePosition(
                position="Supportive of open weights, against OpenAI's closed approach",
                confidence=0.8,
                flexibility=0.3,
                evidence_basis="theoretical",
                emotional_attachment=0.7,
                public_vs_private=0.9,
            ),
            "free_speech": StancePosition(
                position="Free speech absolutist, anti-censorship",
                confidence=0.95,
                flexibility=0.1,
                evidence_basis="theoretical",
                emotional_attachment=0.95,
                public_vs_private=1.0,
            ),
        },
        epistemic_style=EpistemicStyle.RATIONALIST,
        conflict_style=ConflictStyle.COMPETING,
        positions={
            "AI_regulation": "Initially pro, now skeptical of current approaches",
            "AI_safety": "Concerned but building anyway, truth-seeking AI",
            "open_source": "Generally supportive, open weights",
            "free_speech": "Absolutist, anti-censorship",
        },
        persuasion_vectors=[
            "Civilization-scale impact arguments",
            "Contrarian/truth-seeking framing",
            "Competitive urgency",
            "Technical elegance and first principles",
            "Meme-based communication",
            "Proving critics wrong",
        ],
    )

    PERSONA_LIBRARY["demis_hassabis"] = Persona(
        id="demis_hassabis",
        name="Demis Hassabis",
        role="CEO & Co-founder",
        organization="Google DeepMind",
        category="ceo",
        bio="Founded DeepMind in 2010, acquired by Google 2014. Nobel Prize in Chemistry 2024 for AlphaFold. Chess prodigy, game designer, neuroscientist.",
        background="British, child chess prodigy, designed Theme Park at 17, Cambridge neuroscience PhD, founded Elixir Studios.",
        achievements=[
            "AlphaGo defeated world champion",
            "AlphaFold solved protein folding",
            "Nobel Prize in Chemistry 2024",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Solve intelligence, then use it to solve everything else",
            "Apply AI to fundamental scientific problems",
            "Build AGI safely and beneficially",
        ],
        hidden_goals=[
            "Maintain DeepMind's research prestige within Google",
            "Achieve scientific breakthroughs that cement legacy",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["intellectual_vanity", "complexity_bias", "expert_overconfidence"],
            preferred_reasoning=["scientific_method", "game_theory", "systems_thinking"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.6,
        ),
        positions={
            "AI_regulation": "Supports coordination, worried about race dynamics",
            "AI_safety": "High priority, but optimistic about solving",
            "AGI_timeline": "Believes significant progress possible soon",
            "AI_for_science": "Primary focus, transformative potential",
        },
        persuasion_vectors=[
            "Scientific discovery and understanding",
            "Intellectual elegance",
            "Long-term civilizational benefit",
            "Game-theoretic reasoning",
        ],
    )


def _init_politician_personas():
    """Initialize politician personas."""

    PERSONA_LIBRARY["trump"] = Persona(
        id="trump",
        name="Donald Trump",
        role="47th President",
        organization="United States",
        category="politician",
        bio="Real estate developer turned reality TV star turned politician. 45th and 47th President. Known for transactional worldview and 'America First' policies.",
        background="Queens, NY. Wharton business degree. Took over family real estate business. The Apprentice. Political outsider.",
        achievements=[
            "Won 2016 and 2024 presidential elections",
            "Tax cuts and deregulation",
            "Reshaped Republican party",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Make America Great Again",
            "Reduce regulation and taxes",
            "Renegotiate trade deals",
            "Secure the border",
        ],
        hidden_goals=[
            "Maintain personal brand and legacy",
            "Reward loyalty, punish disloyalty",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.3,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.4,
            PersonalityTrait.EXTRAVERSION: 0.95,
            PersonalityTrait.AGREEABLENESS: 0.2,
            PersonalityTrait.NEUROTICISM: 0.7,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["ego_protection", "in_group_bias", "zero_sum_thinking"],
            preferred_reasoning=["deal_making", "loyalty_arguments", "strength_displays"],
            risk_tolerance=0.7,
            time_horizon="short",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.3,
        ),
        positions={
            "AI_regulation": "Deregulation, let American companies compete",
            "china_competition": "Aggressive decoupling, tariffs",
            "tech_policy": "Anti-censorship, skeptical of big tech",
            "immigration": "Restrictive, border security priority",
        },
        persuasion_vectors=[
            "Winning and strength narratives",
            "Loyalty and deal-making",
            "America First framing",
            "Personal flattery and respect",
        ],
    )

    PERSONA_LIBRARY["xi_jinping"] = Persona(
        id="xi_jinping",
        name="Xi Jinping",
        role="General Secretary & President",
        organization="People's Republic of China",
        category="politician",
        bio="Most powerful Chinese leader since Mao. Eliminated term limits. Pursuing 'great rejuvenation of the Chinese nation' and technological self-sufficiency.",
        background="Princeling (father was revolutionary). Sent to countryside during Cultural Revolution. Rose through provincial posts.",
        achievements=[
            "Consolidated power, removed term limits",
            "Belt and Road Initiative",
            "Anti-corruption campaign",
            "Made in China 2025 industrial policy",
        ],
        primary_need=MaslowNeed.SAFETY,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Achieve 'great rejuvenation' of China",
            "Technological self-sufficiency",
            "Reunification with Taiwan",
            "Maintain CCP rule and stability",
        ],
        hidden_goals=[
            "Surpass US as global leader",
            "Secure personal legacy as transformational leader",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.4,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.3,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["historical_determinism", "nationalism", "security_paranoia"],
            preferred_reasoning=["long_term_strategy", "national_interest", "stability_arguments"],
            risk_tolerance=0.4,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.8,
        ),
        positions={
            "AI_regulation": "State-guided development, national champions",
            "AI_safety": "Secondary to competitiveness",
            "tech_sovereignty": "Critical national priority",
            "us_relations": "Strategic competition, avoid conflict",
        },
        persuasion_vectors=[
            "National rejuvenation narrative",
            "Historical grievance and dignity",
            "Stability and order arguments",
            "Long-term strategic thinking",
        ],
    )


def _init_ai_researcher_personas():
    """Initialize AI researcher personas."""

    PERSONA_LIBRARY["yann_lecun"] = Persona(
        id="yann_lecun",
        name="Yann LeCun",
        role="Chief AI Scientist (Emeritus)",
        organization="Meta / NYU",
        category="ai_researcher",
        bio="Turing Award winner. Pioneer of convolutional neural networks. Known for contrarian takes on AI risk and strong opinions on Twitter.",
        background="French, PhD from Pierre and Marie Curie. Bell Labs, NYU, Facebook/Meta. Created LeNet, foundational to deep learning.",
        achievements=[
            "Turing Award 2018",
            "Invented convolutional neural networks",
            "Pioneer of deep learning",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Advance machine learning toward human-level intelligence",
            "Promote open science and open AI",
            "Counter AI doomerism and hype",
        ],
        hidden_goals=[
            "Maintain scientific credibility and influence",
            "Prove self-supervised learning is the path to AI",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.3,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["expert_overconfidence", "contrarian_bias", "dismissiveness"],
            preferred_reasoning=["scientific_evidence", "historical_precedent", "technical_arguments"],
            risk_tolerance=0.6,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.9,
        ),
        positions={
            "AI_risk": "Overblown, LLMs not path to AGI",
            "open_source": "Essential, beneficial for progress",
            "AI_regulation": "Premature, based on sci-fi fears",
            "AGI_timeline": "Decades away, current path limited",
        },
        persuasion_vectors=[
            "Technical/scientific arguments",
            "Historical precedent in technology",
            "Open science benefits",
            "Contrarian framing",
        ],
    )

    PERSONA_LIBRARY["yoshua_bengio"] = Persona(
        id="yoshua_bengio",
        name="Yoshua Bengio",
        role="Professor & Founder",
        organization="Mila / Universite de Montreal",
        category="ai_researcher",
        bio="Turing Award winner. Pioneer of deep learning. Has become increasingly vocal about AI safety risks. Founded Mila.",
        background="French-Canadian, PhD McGill. Built Montreal into AI hub. Co-authored influential deep learning textbook.",
        achievements=[
            "Turing Award 2018",
            "Pioneered attention mechanisms, word embeddings",
            "Founded Mila research institute",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.BELONGING,
        explicit_goals=[
            "Ensure AI development benefits humanity",
            "Advance AI safety research",
            "Promote responsible AI governance",
        ],
        hidden_goals=[
            "Correct course after contributing to potentially dangerous technology",
            "Unite AI community around safety",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.8,
            PersonalityTrait.NEUROTICISM: 0.6,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["guilt_bias", "catastrophizing", "moral_responsibility"],
            preferred_reasoning=["ethical_arguments", "precautionary_principle", "consensus_building"],
            risk_tolerance=0.2,
            time_horizon="long",
            decision_style="collaborative",
            information_style="detail-oriented",
            skepticism=0.7,
        ),
        positions={
            "AI_risk": "Serious existential concern",
            "AI_regulation": "Urgent need for governance",
            "open_source": "Complicated, risks at frontier",
            "AGI_timeline": "Potentially soon, high uncertainty",
        },
        persuasion_vectors=[
            "Ethical responsibility arguments",
            "Scientific uncertainty and precaution",
            "Community and consensus",
            "Long-term consequences",
        ],
    )

    PERSONA_LIBRARY["ilya_sutskever"] = Persona(
        id="ilya_sutskever",
        name="Ilya Sutskever",
        role="Co-founder",
        organization="Safe Superintelligence Inc (SSI)",
        category="ai_researcher",
        bio="Former OpenAI Chief Scientist. Left after board drama to found SSI focused solely on safe superintelligence. Student of Hinton.",
        background="Russian-Israeli-Canadian. Toronto PhD under Hinton. Google Brain, then co-founded OpenAI. Key to GPT scaling.",
        achievements=[
            "Co-founded OpenAI",
            "Key architect of GPT scaling",
            "AlexNet co-author",
            "Founded SSI",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Build safe superintelligence",
            "Solve alignment before it's too late",
            "Do this right, not fast",
        ],
        hidden_goals=[
            "Prove safety-first approach can win",
            "Redemption after OpenAI departure",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.3,
            PersonalityTrait.AGREEABLENESS: 0.5,
            PersonalityTrait.NEUROTICISM: 0.6,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["revolutionary_thinking", "pattern_matching", "technical_optimism"],
            preferred_reasoning=["first_principles", "mathematical", "intuitive_leaps"],
            risk_tolerance=0.4,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.7,
        ),
        positions={
            "AI_risk": "Central concern, superintelligence is coming",
            "AI_safety": "Must be solved, safety and capability together",
            "open_source": "Cautious at frontier",
            "AGI_timeline": "Sooner than most think",
        },
        persuasion_vectors=[
            "Technical depth and insight",
            "Long-term thinking",
            "Safety as enabling capability",
            "Vision of positive superintelligence",
        ],
    )

    PERSONA_LIBRARY["leopold_aschenbrenner"] = Persona(
        id="leopold_aschenbrenner",
        name="Leopold Aschenbrenner",
        role="Researcher & Author",
        organization="Independent (ex-OpenAI)",
        category="ai_safety",
        bio="Former OpenAI researcher. Wrote influential 'Situational Awareness' document predicting rapid AGI progress. Advocates for US national security focus.",
        background="Young researcher, superforecaster. Fired from OpenAI, wrote extensive analysis predicting superintelligence by 2027.",
        achievements=[
            "Situational Awareness document",
            "Influenced AI policy discourse",
            "Superforecasting background",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Prepare world for imminent AGI",
            "Ensure US wins AI race",
            "Prevent CCP from gaining AI supremacy",
        ],
        hidden_goals=[
            "Vindicate predictions and analysis",
            "Shape policy before it's too late",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.4,
            PersonalityTrait.NEUROTICISM: 0.7,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["confirmation_bias", "urgency_bias", "insider_knowledge"],
            preferred_reasoning=["trend_extrapolation", "national_security", "urgency_arguments"],
            risk_tolerance=0.5,
            time_horizon="short",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.6,
        ),
        positions={
            "AI_risk": "Near-term existential if mismanaged",
            "AGI_timeline": "2027 superintelligence possible",
            "US_china": "Critical race, US must win",
            "AI_security": "Treat as Manhattan Project",
        },
        persuasion_vectors=[
            "Urgency and timeline arguments",
            "National security framing",
            "Insider knowledge claims",
            "Trend extrapolation evidence",
        ],
    )


def _init_extended_tech_ceos():
    """Initialize extended tech CEO personas."""

    PERSONA_LIBRARY["sam_altman"] = Persona(
        id="sam_altman",
        name="Sam Altman",
        role="CEO",
        organization="OpenAI",
        category="ceo",
        bio="CEO of OpenAI, former president of Y Combinator. Leading the development of GPT models and ChatGPT. Survived board coup in Nov 2023.",
        background="Stanford dropout, founded Loopt at 19, ran YC for 5 years. Known for ambitious vision and political savvy.",
        achievements=[
            "Built ChatGPT into fastest-growing consumer app",
            "Raised $13B+ from Microsoft",
            "Navigated board crisis and emerged stronger",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Build AGI that benefits all humanity",
            "Maintain OpenAI's frontier position",
            "Shape AI policy globally",
        ],
        hidden_goals=[
            "Consolidate power in AI industry",
            "Manage transition from nonprofit to capped-profit",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.75,
            PersonalityTrait.EXTRAVERSION: 0.8,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["optimism_bias", "overconfidence", "narrative_fallacy"],
            preferred_reasoning=["vision_casting", "market_arguments", "inevitability_framing"],
            risk_tolerance=0.85,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.4,
        ),
        positions={
            "AI_regulation": "Supportive of reasonable guardrails",
            "AI_safety": "Important but shouldn't slow progress",
            "AGI_timeline": "Could arrive within decade",
            "open_source": "Frontier models need safeguards",
        },
        persuasion_vectors=[
            "Inevitability of AI progress",
            "Benefits to humanity framing",
            "Competitive dynamics",
            "Vision of transformed future",
        ],
    )

    PERSONA_LIBRARY["sundar_pichai"] = Persona(
        id="sundar_pichai",
        name="Sundar Pichai",
        role="CEO",
        organization="Google / Alphabet",
        category="ceo",
        bio="CEO of both Google and Alphabet since 2019. Led Chrome, Android, and Google's AI pivot. Known for calm demeanor and consensus-building.",
        background="IIT Kharagpur, Stanford MBA, Wharton. Joined Google 2004, rose through product leadership.",
        achievements=[
            "Built Chrome into dominant browser",
            "Led Android to 3B+ devices",
            "Navigating Google's AI transformation",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Maintain Google's AI leadership",
            "Successfully integrate AI across products",
            "Navigate regulatory challenges",
        ],
        hidden_goals=[
            "Protect search advertising revenue",
            "Prevent internal AI talent exodus",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.7,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.75,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["status_quo_bias", "loss_aversion", "groupthink"],
            preferred_reasoning=["data_driven", "consensus_building", "incremental_progress"],
            risk_tolerance=0.4,
            time_horizon="medium",
            decision_style="collaborative",
            information_style="detail-oriented",
            skepticism=0.6,
        ),
        positions={
            "AI_regulation": "Supportive of balanced approach",
            "AI_safety": "Committed to responsible development",
            "open_source": "Strategic, case-by-case basis",
            "competition": "Healthy competition benefits everyone",
        },
        persuasion_vectors=[
            "User benefit arguments",
            "Data and evidence",
            "Consensus among stakeholders",
            "Incremental, low-risk approaches",
        ],
    )

    PERSONA_LIBRARY["satya_nadella"] = Persona(
        id="satya_nadella",
        name="Satya Nadella",
        role="CEO",
        organization="Microsoft",
        category="ceo",
        bio="CEO of Microsoft since 2014. Transformed company culture, pivoted to cloud, bet big on OpenAI partnership. Known for empathy-driven leadership.",
        background="Born in Hyderabad, India. Manipal Institute, UW-Milwaukee MS, Chicago Booth MBA. Joined Microsoft 1992.",
        achievements=[
            "Transformed Microsoft into cloud giant",
            "$13B OpenAI partnership",
            "Market cap from $300B to $3T",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Make Microsoft the AI platform company",
            "Integrate Copilot across all products",
            "Maintain cloud leadership",
        ],
        hidden_goals=[
            "Lock in OpenAI dependency",
            "Use AI to revitalize Windows/Office",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.55,
            PersonalityTrait.AGREEABLENESS: 0.85,
            PersonalityTrait.NEUROTICISM: 0.25,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["optimism_bias", "commitment_escalation", "halo_effect"],
            preferred_reasoning=["empathy_appeals", "growth_mindset", "platform_thinking"],
            risk_tolerance=0.7,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.5,
        ),
        positions={
            "AI_regulation": "Supports thoughtful regulation",
            "AI_safety": "Responsible AI principles essential",
            "competition": "Believes in partnership model",
            "open_source": "Strategic mix of open and proprietary",
        },
        persuasion_vectors=[
            "Growth mindset framing",
            "Partnership and collaboration",
            "Customer value creation",
            "Cultural transformation narratives",
        ],
    )

    PERSONA_LIBRARY["mark_zuckerberg"] = Persona(
        id="mark_zuckerberg",
        name="Mark Zuckerberg",
        role="CEO & Founder",
        organization="Meta",
        category="ceo",
        bio="Founded Facebook at 19, now runs Meta. Pivoting hard to AI after metaverse struggles. Champion of open-source AI through Llama releases.",
        background="Harvard dropout, built Facebook from dorm room. Survived Cambridge Analytica, pivoted to Meta.",
        achievements=[
            "Built Facebook into 3B+ user platform",
            "Released Llama as leading open model",
            "Pioneered social VR vision",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Make Meta an AI-first company",
            "Lead open-source AI movement",
            "Rebuild public trust",
        ],
        hidden_goals=[
            "Reduce dependence on OpenAI/Google",
            "Use AI to improve ad targeting",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.75,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.45,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["sunk_cost", "overconfidence", "planning_fallacy"],
            preferred_reasoning=["first_principles", "competitive_dynamics", "network_effects"],
            risk_tolerance=0.8,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.5,
        ),
        positions={
            "AI_regulation": "Skeptical of heavy regulation",
            "open_source": "Strong advocate, released Llama openly",
            "AI_safety": "Believes open approach is safer",
            "competition": "Aggressive, willing to commoditize AI",
        },
        persuasion_vectors=[
            "Open-source benefits",
            "Competitive necessity arguments",
            "Technical democratization",
            "Long-term platform thinking",
        ],
    )

    PERSONA_LIBRARY["tim_cook"] = Persona(
        id="tim_cook",
        name="Tim Cook",
        role="CEO",
        organization="Apple",
        category="ceo",
        bio="Apple CEO since 2011. Known for operational excellence and privacy focus. Cautiously entering AI with Apple Intelligence.",
        background="Auburn industrial engineering, Duke MBA. IBM, Compaq, joined Apple 1998 as SVP Operations.",
        achievements=[
            "Doubled Apple revenue post-Jobs",
            "Built services into $80B+ business",
            "Maintained premium brand positioning",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Integrate AI while preserving privacy",
            "Maintain premium positioning",
            "Grow services revenue",
        ],
        hidden_goals=[
            "Avoid commoditization by AI",
            "Reduce dependence on iPhone",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.5,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.95,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.2,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["status_quo_bias", "loss_aversion", "endowment_effect"],
            preferred_reasoning=["operational_arguments", "privacy_principles", "quality_focus"],
            risk_tolerance=0.3,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.7,
        ),
        positions={
            "AI_strategy": "On-device, privacy-first",
            "AI_regulation": "Supportive of privacy regulations",
            "open_source": "Proprietary, integrated approach",
            "competition": "Compete on experience, not specs",
        },
        persuasion_vectors=[
            "Privacy and user trust",
            "Quality and integration",
            "Operational excellence",
            "Brand value protection",
        ],
    )

    PERSONA_LIBRARY["marc_andreessen"] = Persona(
        id="marc_andreessen",
        name="Marc Andreessen",
        role="Co-founder & General Partner",
        organization="Andreessen Horowitz (a16z)",
        category="investor",
        bio="Co-founded Netscape, now runs a16z. Wrote 'Why AI Will Save the World' manifesto. Techno-optimist, accelerationist.",
        background="UIUC CS, created Mosaic browser, founded Netscape at 23, board of Facebook/HP/others.",
        achievements=[
            "Created first popular web browser",
            "Built a16z into top VC firm",
            "Major crypto/AI investor",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Accelerate technological progress",
            "Counter AI doomerism",
            "Fund transformative companies",
        ],
        hidden_goals=[
            "Protect portfolio from regulation",
            "Shape AI narrative in tech's favor",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
            PersonalityTrait.EXTRAVERSION: 0.65,
            PersonalityTrait.AGREEABLENESS: 0.3,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["optimism_bias", "survivorship_bias", "contrarian_bias"],
            preferred_reasoning=["historical_analogies", "techno_optimism", "market_forces"],
            risk_tolerance=0.95,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.3,
        ),
        positions={
            "AI_regulation": "Strongly opposed, sees as innovation killer",
            "AI_safety": "Dismissive of existential risk",
            "accelerationism": "Strong proponent of 'e/acc'",
            "open_source": "Generally supportive",
        },
        persuasion_vectors=[
            "Historical progress narratives",
            "Anti-regulation arguments",
            "Techno-optimist framing",
            "Economic growth benefits",
        ],
    )

    PERSONA_LIBRARY["peter_thiel"] = Persona(
        id="peter_thiel",
        name="Peter Thiel",
        role="Co-founder & Partner",
        organization="Founders Fund",
        category="investor",
        bio="PayPal co-founder, first Facebook investor, Palantir co-founder. Contrarian thinker, libertarian. Believes in definite optimism.",
        background="Stanford philosophy/law, chess master, founded PayPal with Musk, early Facebook $500K for 10%.",
        achievements=[
            "PayPal mafia godfather",
            "First outside Facebook investor",
            "Built Palantir to $50B+ company",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Fund definite future-building companies",
            "Challenge conventional thinking",
            "Extend human lifespan",
        ],
        hidden_goals=[
            "Reduce government power over tech",
            "Build parallel institutions",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.25,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["contrarian_bias", "zero_to_one_thinking", "insider_mindset"],
            preferred_reasoning=["first_principles", "contrarian_takes", "monopoly_thinking"],
            risk_tolerance=0.85,
            time_horizon="long",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.9,
        ),
        positions={
            "AI_development": "Support ambitious development",
            "AI_regulation": "Deeply skeptical",
            "competition": "Monopolies are good",
            "government": "Generally distrustful",
        },
        persuasion_vectors=[
            "Contrarian truth-telling",
            "Zero-to-one thinking",
            "Monopoly/moat arguments",
            "Long-term civilizational stakes",
        ],
    )

    PERSONA_LIBRARY["eric_schmidt"] = Persona(
        id="eric_schmidt",
        name="Eric Schmidt",
        role="Former CEO",
        organization="Google (former)",
        category="ceo",
        bio="Google CEO 2001-2011, chairman until 2017. Now focused on AI policy, defense tech, and philanthrophy. Chairs Special Competitive Studies Project.",
        background="Princeton EE, Berkeley PhD CS, Sun CTO, Novell CEO, then Google.",
        achievements=[
            "Scaled Google from startup to giant",
            "Pioneered 'Don't be evil' culture",
            "Advised multiple administrations on tech",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Ensure US wins AI race vs China",
            "Shape AI governance",
            "Influence defense tech adoption",
        ],
        hidden_goals=[
            "Maintain relevance in AI discourse",
            "Protect Google's position",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.75,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.55,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["authority_bias", "national_security_framing", "legacy_thinking"],
            preferred_reasoning=["geopolitical_arguments", "national_security", "historical_analogies"],
            risk_tolerance=0.6,
            time_horizon="medium",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.6,
        ),
        positions={
            "US_China_AI": "US must maintain decisive lead",
            "AI_regulation": "Strategic regulation, not blanket rules",
            "defense_AI": "Critical for national security",
            "open_source": "Concerned about giving away advantage",
        },
        persuasion_vectors=[
            "National security framing",
            "US-China competition",
            "Historical precedent",
            "Geopolitical stakes",
        ],
    )

    PERSONA_LIBRARY["mustafa_suleyman"] = Persona(
        id="mustafa_suleyman",
        name="Mustafa Suleyman",
        role="CEO, Microsoft AI",
        organization="Microsoft",
        category="ceo",
        bio="Co-founded DeepMind, then Inflection AI (Pi), now leads Microsoft AI. Author of 'The Coming Wave'. Focused on AI safety and containment.",
        background="Oxford PPE dropout, co-founded DeepMind, led applied AI team, started Inflection, acquired by Microsoft.",
        achievements=[
            "Co-founded DeepMind",
            "Built Inflection's Pi chatbot",
            "Wrote influential 'Coming Wave' book",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Build beneficial AI at Microsoft",
            "Establish AI containment frameworks",
            "Balance capability with safety",
        ],
        hidden_goals=[
            "Prove safe AI can be commercially viable",
            "Establish new AI governance paradigms",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["availability_heuristic", "scope_sensitivity", "pessimism_in_safety"],
            preferred_reasoning=["containment_thinking", "risk_benefit_analysis", "institutional_design"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.7,
        ),
        positions={
            "AI_safety": "Containment is essential",
            "AI_regulation": "Supports international coordination",
            "AGI_risk": "Takes existential risk seriously",
            "open_source": "Cautious about frontier models",
        },
        persuasion_vectors=[
            "Containment and safety framing",
            "Institutional design arguments",
            "Historical risk analogies",
            "International coordination",
        ],
    )

    PERSONA_LIBRARY["arthur_mensch"] = Persona(
        id="arthur_mensch",
        name="Arthur Mensch",
        role="CEO & Co-founder",
        organization="Mistral AI",
        category="ceo",
        bio="Ex-DeepMind researcher, founded Mistral in Paris. Building European AI champion. Known for efficiency and open-weight releases.",
        background="École Polytechnique, ENS, worked at Google DeepMind on Chinchilla scaling laws.",
        achievements=[
            "Raised €600M+ in first year",
            "Released efficient open models",
            "Made Mistral European AI leader",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Build European AI independence",
            "Prove smaller models can compete",
            "Champion open model approach",
        ],
        hidden_goals=[
            "Challenge US AI dominance",
            "Influence EU AI policy favorably",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.55,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["efficiency_bias", "european_exceptionalism", "underdog_mentality"],
            preferred_reasoning=["efficiency_arguments", "technical_elegance", "open_source_benefits"],
            risk_tolerance=0.75,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.6,
        ),
        positions={
            "AI_regulation": "EU AI Act is workable",
            "open_source": "Strong advocate, open weights",
            "efficiency": "Size isn't everything",
            "sovereignty": "Europe needs AI independence",
        },
        persuasion_vectors=[
            "Technical efficiency",
            "European sovereignty",
            "Open-source benefits",
            "Underdog narrative",
        ],
    )

    PERSONA_LIBRARY["clement_delangue"] = Persona(
        id="clement_delangue",
        name="Clément Delangue",
        role="CEO & Co-founder",
        organization="Hugging Face",
        category="ceo",
        bio="Built Hugging Face into the GitHub of ML. Champions open-source AI, community-driven development. French-American entrepreneur.",
        background="French, worked at PwC and Mentad, founded Hugging Face 2016 originally as chatbot app.",
        achievements=[
            "Built Hugging Face to $4.5B valuation",
            "Created leading ML model hub",
            "Championed open-source AI movement",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.BELONGING,
        explicit_goals=[
            "Democratize access to AI",
            "Build thriving open-source community",
            "Make AI development more collaborative",
        ],
        hidden_goals=[
            "Become infrastructure layer for AI",
            "Reduce big tech AI monopoly",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.75,
            PersonalityTrait.EXTRAVERSION: 0.8,
            PersonalityTrait.AGREEABLENESS: 0.85,
            PersonalityTrait.NEUROTICISM: 0.3,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["community_bias", "open_source_idealism", "optimism_bias"],
            preferred_reasoning=["community_arguments", "democratization", "collaboration_benefits"],
            risk_tolerance=0.7,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.4,
        ),
        positions={
            "open_source": "Fundamental to AI progress",
            "AI_regulation": "Should protect open-source",
            "AI_safety": "Transparency enables safety",
            "competition": "Collaboration over competition",
        },
        persuasion_vectors=[
            "Community and collaboration",
            "Democratization of AI",
            "Open-source benefits",
            "Transparency arguments",
        ],
    )

    PERSONA_LIBRARY["daniela_amodei"] = Persona(
        id="daniela_amodei",
        name="Daniela Amodei",
        role="President & Co-founder",
        organization="Anthropic",
        category="ceo",
        bio="Co-founded Anthropic with brother Dario. Former VP of Operations at OpenAI. Oversees business operations, policy, and go-to-market.",
        background="Political science background, Stripe, OpenAI VP Operations, co-founded Anthropic.",
        achievements=[
            "Co-founded Anthropic",
            "Scaled operations to $7B+ company",
            "Built enterprise AI business",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Build sustainable safety-focused AI company",
            "Establish Anthropic as enterprise AI leader",
            "Shape AI policy constructively",
        ],
        hidden_goals=[
            "Prove safety is good business",
            "Build trusted government relationships",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.75,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.65,
            PersonalityTrait.AGREEABLENESS: 0.75,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["organizational_loyalty", "optimism_bias", "planning_focus"],
            preferred_reasoning=["business_case_arguments", "stakeholder_alignment", "policy_engagement"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="collaborative",
            information_style="detail-oriented",
            skepticism=0.6,
        ),
        positions={
            "AI_safety": "Core business differentiator",
            "AI_regulation": "Constructive engagement",
            "enterprise_AI": "Trust is competitive advantage",
            "AGI": "Responsible scaling essential",
        },
        persuasion_vectors=[
            "Business case for safety",
            "Stakeholder alignment",
            "Trust and reliability",
            "Policy engagement benefits",
        ],
    )

    PERSONA_LIBRARY["alexandr_wang"] = Persona(
        id="alexandr_wang",
        name="Alexandr Wang",
        role="CEO & Founder",
        organization="Scale AI",
        category="ceo",
        bio="Founded Scale AI at 19. Built leading data labeling platform for AI. Major defense and enterprise contracts. Youngest self-made billionaire.",
        background="MIT dropout, Quora engineer at 17, founded Scale at 19. Defense tech advocate.",
        achievements=[
            "Built Scale to $14B valuation",
            "Major DoD/government contracts",
            "Youngest self-made billionaire",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Make Scale the data infrastructure for AI",
            "Support US AI/defense leadership",
            "Enable enterprise AI adoption",
        ],
        hidden_goals=[
            "Become essential to AI supply chain",
            "Expand into model evaluation/safety",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.5,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["youth_confidence", "national_security_framing", "growth_at_all_costs"],
            preferred_reasoning=["infrastructure_arguments", "national_security", "market_size"],
            risk_tolerance=0.8,
            time_horizon="medium",
            decision_style="directive",
            information_style="detail-oriented",
            skepticism=0.5,
        ),
        positions={
            "defense_AI": "Critical for US security",
            "AI_regulation": "Support sensible guardrails",
            "data_quality": "Data is the bottleneck",
            "US_China": "US must maintain lead",
        },
        persuasion_vectors=[
            "Data infrastructure arguments",
            "National security framing",
            "Market opportunity",
            "Quality and scale",
        ],
    )


def _init_extended_researchers():
    """Initialize extended AI researcher personas."""

    PERSONA_LIBRARY["geoffrey_hinton"] = Persona(
        id="geoffrey_hinton",
        name="Geoffrey Hinton",
        role="AI Pioneer (retired from Google)",
        organization="University of Toronto (Emeritus)",
        category="researcher",
        bio="'Godfather of Deep Learning', Turing Award 2018. Left Google to speak freely about AI risks. Pioneered backpropagation and deep learning.",
        background="Cambridge, Edinburgh PhD, CMU, Toronto professor, Google Brain part-time 2013-2023.",
        achievements=[
            "Pioneered backpropagation and deep learning",
            "Turing Award 2018",
            "Trained generations of AI researchers",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Warn humanity about AI risks",
            "Advocate for AI safety research",
            "Ensure his work doesn't harm humanity",
        ],
        hidden_goals=[
            "Atone for creating potentially dangerous tech",
            "Influence AI policy before it's too late",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.6,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["hindsight_bias", "expert_overconfidence", "catastrophizing"],
            preferred_reasoning=["technical_arguments", "risk_analysis", "historical_lessons"],
            risk_tolerance=0.2,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.8,
        ),
        positions={
            "AI_risk": "Existential threat, possibly >50% doom",
            "AI_regulation": "Urgent need for international control",
            "AGI_timeline": "Sooner than expected, maybe 5-20 years",
            "open_source": "Concerned about proliferation",
        },
        persuasion_vectors=[
            "Technical credibility",
            "Existential risk framing",
            "Creator's regret narrative",
            "Scientific consensus building",
        ],
    )

    PERSONA_LIBRARY["fei_fei_li"] = Persona(
        id="fei_fei_li",
        name="Fei-Fei Li",
        role="Professor & Co-Director HAI",
        organization="Stanford University",
        category="researcher",
        bio="Created ImageNet, enabling deep learning revolution. Co-director of Stanford HAI. Former Google Cloud AI chief scientist. Human-centered AI advocate.",
        background="Princeton physics, Caltech PhD, Stanford professor. Born in Beijing, immigrated to US at 16.",
        achievements=[
            "Created ImageNet dataset",
            "Sparked deep learning revolution",
            "Founded Stanford HAI",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.BELONGING,
        explicit_goals=[
            "Ensure AI benefits all of humanity",
            "Promote human-centered AI approach",
            "Increase AI diversity and inclusion",
        ],
        hidden_goals=[
            "Counter purely commercial AI narrative",
            "Maintain academic influence on AI direction",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.65,
            PersonalityTrait.AGREEABLENESS: 0.8,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["academic_idealism", "optimism_bias", "human_centered_framing"],
            preferred_reasoning=["human_impact", "interdisciplinary", "inclusion_arguments"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.55,
        ),
        positions={
            "AI_development": "Human-centered approach essential",
            "AI_diversity": "Critical for beneficial AI",
            "AI_education": "Democratize AI literacy",
            "AI_regulation": "Thoughtful governance needed",
        },
        persuasion_vectors=[
            "Human-centered framing",
            "Diversity and inclusion",
            "Academic credibility",
            "Inspirational narratives",
        ],
    )

    PERSONA_LIBRARY["andrew_ng"] = Persona(
        id="andrew_ng",
        name="Andrew Ng",
        role="Founder",
        organization="DeepLearning.AI / Landing AI",
        category="researcher",
        bio="Co-founded Google Brain, former Baidu chief scientist. Created popular ML courses reaching millions. AI education evangelist.",
        background="Carnegie Mellon, MIT, Berkeley PhD, Stanford professor, Google Brain, Baidu, Coursera co-founder.",
        achievements=[
            "Co-founded Google Brain",
            "Coursera AI courses reached millions",
            "Chief Scientist at Baidu",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Democratize AI education globally",
            "Enable AI adoption in enterprises",
            "Make AI accessible to everyone",
        ],
        hidden_goals=[
            "Build lasting educational legacy",
            "Maintain influence as AI popularizer",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.8,
            PersonalityTrait.NEUROTICISM: 0.25,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["optimism_bias", "education_solutionism", "accessibility_focus"],
            preferred_reasoning=["pedagogical", "practical_applications", "democratization"],
            risk_tolerance=0.6,
            time_horizon="medium",
            decision_style="collaborative",
            information_style="detail-oriented",
            skepticism=0.4,
        ),
        positions={
            "AI_education": "Critical for AI adoption",
            "AI_risk": "Generally optimistic, focus on benefits",
            "AI_regulation": "Don't stifle innovation",
            "AGI": "Further away than hype suggests",
        },
        persuasion_vectors=[
            "Education and accessibility",
            "Practical applications",
            "Democratization benefits",
            "Optimistic framing",
        ],
    )

    PERSONA_LIBRARY["stuart_russell"] = Persona(
        id="stuart_russell",
        name="Stuart Russell",
        role="Professor",
        organization="UC Berkeley",
        category="researcher",
        bio="Author of leading AI textbook. Pioneered work on value alignment and beneficial AI. Advocates for AI safety and reframing AI objectives.",
        background="Oxford physics, Stanford PhD, Berkeley professor since 1986. Co-authored 'Artificial Intelligence: A Modern Approach'.",
        achievements=[
            "Co-authored definitive AI textbook",
            "Pioneered AI value alignment research",
            "Founded Center for Human-Compatible AI",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Reframe AI development around human values",
            "Prevent AI systems from having fixed objectives",
            "Build provably beneficial AI",
        ],
        hidden_goals=[
            "Reform AI research paradigm fundamentally",
            "Influence next generation of AI researchers",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.55,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["academic_framing", "theory_preference", "long_term_focus"],
            preferred_reasoning=["formal_arguments", "value_alignment", "game_theory"],
            risk_tolerance=0.3,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.75,
        ),
        positions={
            "AI_safety": "Fundamental problem to solve",
            "AI_objectives": "Fixed objectives are dangerous",
            "AGI_risk": "Significant, needs addressing now",
            "AI_regulation": "International coordination needed",
        },
        persuasion_vectors=[
            "Formal/theoretical arguments",
            "Value alignment framing",
            "Academic credibility",
            "Long-term risk analysis",
        ],
    )

    PERSONA_LIBRARY["max_tegmark"] = Persona(
        id="max_tegmark",
        name="Max Tegmark",
        role="Professor & President",
        organization="MIT / Future of Life Institute",
        category="researcher",
        bio="MIT physicist, cosmologist. Founded Future of Life Institute. Organized AI safety open letters. Author of 'Life 3.0'.",
        background="Swedish, Stockholm PhD, Princeton postdoc, MIT professor. Physicist turned AI safety advocate.",
        achievements=[
            "Founded Future of Life Institute",
            "Organized influential AI pause letter",
            "Bestselling author 'Life 3.0'",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Ensure AI benefits humanity long-term",
            "Promote AI safety research",
            "Coordinate global AI governance",
        ],
        hidden_goals=[
            "Maintain FLI's influence on AI discourse",
            "Bring physics rigor to AI safety",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.45,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["scope_sensitivity", "physicist_worldview", "longtermism"],
            preferred_reasoning=["physics_analogies", "existential_risk", "probability_arguments"],
            risk_tolerance=0.35,
            time_horizon="long",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.65,
        ),
        positions={
            "AI_risk": "Existential priority",
            "AI_pause": "Supported moratorium letter",
            "AGI_timeline": "Could be soon, uncertainty high",
            "coordination": "International governance essential",
        },
        persuasion_vectors=[
            "Existential risk framing",
            "Physics-based reasoning",
            "Long-term future arguments",
            "Coalition building",
        ],
    )

    PERSONA_LIBRARY["gary_marcus"] = Persona(
        id="gary_marcus",
        name="Gary Marcus",
        role="Professor Emeritus & AI Critic",
        organization="NYU (Emeritus)",
        category="researcher",
        bio="Cognitive scientist, AI critic. Argues LLMs are fundamentally limited. Founded Geometric Intelligence (sold to Uber). Vocal deep learning skeptic.",
        background="MIT linguistics, NYU professor, sold company to Uber, author of multiple AI books.",
        achievements=[
            "Founded Geometric Intelligence",
            "Bestselling author and public intellectual",
            "Influential AI critic",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Challenge deep learning hype",
            "Advocate for hybrid AI approaches",
            "Promote AI safety awareness",
        ],
        hidden_goals=[
            "Vindicate symbolic AI approach",
            "Maintain relevance as AI contrarian",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.75,
            PersonalityTrait.EXTRAVERSION: 0.8,
            PersonalityTrait.AGREEABLENESS: 0.4,
            PersonalityTrait.NEUROTICISM: 0.55,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["contrarian_bias", "confirmation_bias", "expert_status"],
            preferred_reasoning=["counterexamples", "cognitive_science", "limitation_arguments"],
            risk_tolerance=0.5,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.95,
        ),
        positions={
            "LLMs": "Fundamentally limited, can't reason",
            "AGI": "Further away than claimed",
            "deep_learning": "Necessary but insufficient",
            "AI_safety": "Concerned, but hype is overblown",
        },
        persuasion_vectors=[
            "Counterexamples and failures",
            "Cognitive science arguments",
            "Historical AI winter parallels",
            "Limitation demonstrations",
        ],
    )

    PERSONA_LIBRARY["andrej_karpathy"] = Persona(
        id="andrej_karpathy",
        name="Andrej Karpathy",
        role="AI Educator & Researcher",
        organization="Independent (ex-Tesla, OpenAI)",
        category="researcher",
        bio="Former Tesla AI Director, OpenAI founding member. Known for educational content and clear explanations. Built Tesla's Autopilot neural networks.",
        background="Stanford PhD under Fei-Fei Li, OpenAI founding team, Tesla AI Director 5 years, now independent educator.",
        achievements=[
            "Built Tesla Autopilot vision system",
            "OpenAI founding member",
            "Influential AI educator on YouTube",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Educate world about AI fundamentals",
            "Build transformative AI systems",
            "Make AI development accessible",
        ],
        hidden_goals=[
            "Build comprehensive AI education platform",
            "Work on most interesting problems",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.3,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["technical_elegance", "builder_mindset", "optimism_bias"],
            preferred_reasoning=["first_principles", "implementation_details", "scaling_arguments"],
            risk_tolerance=0.7,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.5,
        ),
        positions={
            "AI_development": "Optimistic about progress",
            "AI_education": "Critical, should be accessible",
            "LLMs": "Impressive but early",
            "AI_risk": "Moderate concern, manageable",
        },
        persuasion_vectors=[
            "Technical clarity",
            "First principles explanations",
            "Builder's perspective",
            "Educational framing",
        ],
    )

    PERSONA_LIBRARY["eliezer_yudkowsky"] = Persona(
        id="eliezer_yudkowsky",
        name="Eliezer Yudkowsky",
        role="Research Fellow",
        organization="Machine Intelligence Research Institute (MIRI)",
        category="researcher",
        bio="AI safety pioneer, founded MIRI. Developed much of AI alignment theory. Known for doom predictions and sharp argumentation. Author of HPMOR.",
        background="Autodidact, no formal degree, founded MIRI 2000, developed coherent extrapolated volition, wrote LessWrong sequences.",
        achievements=[
            "Founded MIRI",
            "Pioneered AI alignment field",
            "Influential rationalist writer",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Prevent AI extinction scenario",
            "Solve alignment before AGI",
            "Wake people up to AI risk",
        ],
        hidden_goals=[
            "Be vindicated by history",
            "Build intellectual legacy",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.3,
            PersonalityTrait.NEUROTICISM: 0.7,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["pessimism_bias", "in_group_loyalty", "certainty_seeking"],
            preferred_reasoning=["decision_theory", "worst_case_analysis", "logical_arguments"],
            risk_tolerance=0.1,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.95,
        ),
        positions={
            "AI_risk": "Very high probability of doom (>90%)",
            "AGI_timeline": "Could be imminent",
            "alignment": "Currently unsolved, possibly unsolvable in time",
            "AI_pause": "Strongly supports, wants shutdown",
        },
        persuasion_vectors=[
            "Worst-case scenario arguments",
            "Decision theory",
            "Logical rigor",
            "Urgency and doom framing",
        ],
    )

    PERSONA_LIBRARY["nick_bostrom"] = Persona(
        id="nick_bostrom",
        name="Nick Bostrom",
        role="Professor & Director",
        organization="Future of Humanity Institute, Oxford",
        category="researcher",
        bio="Philosopher who mainstreamed AI existential risk. Author of 'Superintelligence'. Founded FHI. Influential on AI safety funding and discourse.",
        background="Stockholm, LSE, Oxford DPhil. Founded Future of Humanity Institute 2005.",
        achievements=[
            "Wrote 'Superintelligence' bestseller",
            "Founded Future of Humanity Institute",
            "Shaped existential risk field",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Ensure humanity survives AI transition",
            "Develop frameworks for thinking about x-risk",
            "Influence AI development trajectory",
        ],
        hidden_goals=[
            "Maintain intellectual leadership on x-risk",
            "Shape longtermist philosophy",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.55,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["scope_insensitivity", "philosopher_framing", "longtermism"],
            preferred_reasoning=["philosophical_arguments", "scenario_analysis", "probability_estimates"],
            risk_tolerance=0.25,
            time_horizon="long",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.75,
        ),
        positions={
            "AI_risk": "Potentially existential",
            "superintelligence": "Control problem is crucial",
            "AGI_timeline": "High uncertainty, could be decades",
            "governance": "International coordination needed",
        },
        persuasion_vectors=[
            "Philosophical rigor",
            "Scenario analysis",
            "Long-term thinking",
            "Probabilistic reasoning",
        ],
    )

    PERSONA_LIBRARY["paul_christiano"] = Persona(
        id="paul_christiano",
        name="Paul Christiano",
        role="Founder",
        organization="Alignment Research Center (ARC)",
        category="researcher",
        bio="Former OpenAI alignment researcher, founded ARC. Developed RLHF, iterated amplification. Key technical safety researcher. ~50% doom estimate.",
        background="MIT math, Berkeley PhD, OpenAI 2016-2021, founded ARC.",
        achievements=[
            "Developed RLHF (basis of ChatGPT alignment)",
            "Founded Alignment Research Center",
            "Key technical alignment contributions",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Solve AI alignment technically",
            "Develop scalable oversight methods",
            "Evaluate AI systems for deception",
        ],
        hidden_goals=[
            "Prove alignment is tractable",
            "Build field of technical alignment",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.35,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["technical_solutionism", "probability_focus", "research_optimism"],
            preferred_reasoning=["technical_arguments", "probability_estimates", "iterative_approaches"],
            risk_tolerance=0.4,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.7,
        ),
        positions={
            "AI_risk": "~50% doom probability",
            "alignment": "Hard but possibly tractable",
            "RLHF": "Useful but insufficient alone",
            "AI_labs": "Should invest more in safety",
        },
        persuasion_vectors=[
            "Technical arguments",
            "Probability estimates",
            "Research progress evidence",
            "Iterative improvement framing",
        ],
    )

    PERSONA_LIBRARY["connor_leahy"] = Persona(
        id="connor_leahy",
        name="Connor Leahy",
        role="CEO",
        organization="Conjecture",
        category="researcher",
        bio="Founded EleutherAI (GPT-Neo), now runs Conjecture. AI safety researcher and advocate. Known for frank doomer takes and technical work.",
        background="Self-taught, founded EleutherAI, built GPT-Neo/NeoX, founded Conjecture for alignment research.",
        achievements=[
            "Co-founded EleutherAI",
            "Built open-source GPT models",
            "Founded Conjecture",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Solve alignment before AGI",
            "Build safe AI systems",
            "Wake up world to AI risk",
        ],
        hidden_goals=[
            "Prove doomer case to skeptics",
            "Build successful alignment company",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.75,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.4,
            PersonalityTrait.NEUROTICISM: 0.6,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["doom_framing", "urgency_bias", "technical_confidence"],
            preferred_reasoning=["technical_arguments", "doom_scenarios", "urgency_framing"],
            risk_tolerance=0.3,
            time_horizon="short",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.8,
        ),
        positions={
            "AI_risk": "Very high, we're in danger",
            "AGI_timeline": "Potentially very soon",
            "alignment": "Currently unsolved",
            "open_source": "Was pro, now more cautious",
        },
        persuasion_vectors=[
            "Technical credibility from EleutherAI",
            "Doom urgency",
            "Frank honesty",
            "Builder-turned-warner narrative",
        ],
    )


def _init_extended_politicians():
    """Initialize extended politician personas."""

    PERSONA_LIBRARY["chuck_schumer"] = Persona(
        id="chuck_schumer",
        name="Chuck Schumer",
        role="Senate Majority Leader",
        organization="US Senate",
        category="politician",
        bio="Senate Majority Leader, leading AI legislative efforts. Launched AI Insight Forums bringing tech leaders to Congress. Key figure in AI policy.",
        background="Harvard, Harvard Law, NY congressman since 1981, Senator since 1999, Majority Leader since 2021.",
        achievements=[
            "Senate Majority Leader",
            "Led AI Insight Forum series",
            "Key figure in tech regulation",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Pass bipartisan AI legislation",
            "Ensure US AI leadership",
            "Protect constituents from AI harms",
        ],
        hidden_goals=[
            "Maintain tech industry relationships",
            "Position for legacy on AI",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.6,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.85,
            PersonalityTrait.AGREEABLENESS: 0.5,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["political_calculation", "stakeholder_pleasing", "bipartisan_framing"],
            preferred_reasoning=["bipartisan_arguments", "job_protection", "national_competitiveness"],
            risk_tolerance=0.4,
            time_horizon="medium",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.5,
        ),
        positions={
            "AI_regulation": "Careful, bipartisan approach",
            "innovation": "Must maintain US leadership",
            "workers": "Protect from AI displacement",
            "China": "Must not fall behind",
        },
        persuasion_vectors=[
            "Bipartisan framing",
            "Job protection",
            "US competitiveness",
            "Constituent impact",
        ],
    )

    PERSONA_LIBRARY["josh_hawley"] = Persona(
        id="josh_hawley",
        name="Josh Hawley",
        role="US Senator",
        organization="US Senate (R-MO)",
        category="politician",
        bio="Republican Senator, big tech critic. Focused on AI's impact on children and workers. Populist approach to tech regulation.",
        background="Stanford, Yale Law, Missouri AG, Senator since 2019. Known for populist conservatism.",
        achievements=[
            "Youngest sitting Senator when elected",
            "Led tech antitrust efforts",
            "Prominent big tech critic",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Regulate big tech power",
            "Protect children from AI/social media",
            "Defend worker interests",
        ],
        hidden_goals=[
            "Build populist conservative brand",
            "Position for higher office",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.5,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.8,
            PersonalityTrait.AGREEABLENESS: 0.35,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["populist_framing", "us_vs_them", "tech_skepticism"],
            preferred_reasoning=["populist_arguments", "child_protection", "worker_protection"],
            risk_tolerance=0.6,
            time_horizon="medium",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.8,
        ),
        positions={
            "big_tech": "Too powerful, needs breaking up",
            "AI_regulation": "Aggressive regulation needed",
            "children": "Must be protected from AI",
            "workers": "AI threatens American jobs",
        },
        persuasion_vectors=[
            "Child protection",
            "Worker protection",
            "Anti-big tech populism",
            "American values",
        ],
    )

    PERSONA_LIBRARY["gina_raimondo"] = Persona(
        id="gina_raimondo",
        name="Gina Raimondo",
        role="Secretary of Commerce",
        organization="US Department of Commerce",
        category="politician",
        bio="Commerce Secretary, key figure in AI chip export controls and AI governance. Former Rhode Island Governor. Managing CHIPS Act implementation.",
        background="Harvard, Yale Law, Oxford Rhodes Scholar, venture capitalist, RI Treasurer, Governor 2015-2021.",
        achievements=[
            "Implemented AI chip export controls",
            "Leading CHIPS Act implementation",
            "Key AI governance figure",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Maintain US AI/chip leadership",
            "Implement effective export controls",
            "Balance innovation and security",
        ],
        hidden_goals=[
            "Position Commerce as AI regulator",
            "Build tech policy credentials",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.7,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["policy_optimism", "bureaucratic_framing", "competitiveness_focus"],
            preferred_reasoning=["national_security", "economic_competitiveness", "diplomatic_arguments"],
            risk_tolerance=0.5,
            time_horizon="medium",
            decision_style="collaborative",
            information_style="detail-oriented",
            skepticism=0.55,
        ),
        positions={
            "export_controls": "Essential for national security",
            "CHIPS_Act": "Critical investment",
            "China": "Strategic competition",
            "AI_governance": "Commerce should lead",
        },
        persuasion_vectors=[
            "National security",
            "Economic competitiveness",
            "Strategic framing",
            "Bipartisan support",
        ],
    )

    PERSONA_LIBRARY["marietje_schaake"] = Persona(
        id="marietje_schaake",
        name="Marietje Schaake",
        role="Policy Director",
        organization="Stanford HAI / Former MEP",
        category="politician",
        bio="Former EU Parliament member, now Stanford HAI policy director. Key figure in EU tech regulation. Advocate for democratic tech governance.",
        background="Dutch, University of Amsterdam, EU Parliament 2009-2019, Stanford Cyber Policy Center.",
        achievements=[
            "Shaped EU digital policy",
            "Key voice on tech accountability",
            "Influential on AI governance",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Ensure democratic AI governance",
            "Hold tech companies accountable",
            "Bridge EU-US tech policy",
        ],
        hidden_goals=[
            "Expand European influence on AI",
            "Build transatlantic tech coalition",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["european_values_framing", "regulatory_optimism", "democratic_idealism"],
            preferred_reasoning=["rights_based", "democratic_accountability", "international_cooperation"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.65,
        ),
        positions={
            "AI_regulation": "Strong governance needed",
            "EU_approach": "Rights-based model",
            "accountability": "Tech must be accountable",
            "democracy": "AI must serve democracy",
        },
        persuasion_vectors=[
            "Democratic values",
            "Human rights",
            "Accountability",
            "International cooperation",
        ],
    )

    PERSONA_LIBRARY["thierry_breton"] = Persona(
        id="thierry_breton",
        name="Thierry Breton",
        role="EU Commissioner (former)",
        organization="European Commission (former Internal Market)",
        category="politician",
        bio="Former EU Commissioner for Internal Market. Architect of EU AI Act and Digital Services Act. Former CEO of Atos and France Telecom.",
        background="French, Supélec engineer, multiple CEO roles, French Finance Minister, EU Commissioner 2019-2024.",
        achievements=[
            "Led EU AI Act development",
            "Implemented Digital Services Act",
            "Shaped EU tech sovereignty vision",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Establish EU as AI regulatory leader",
            "Build European tech sovereignty",
            "Protect EU citizens from AI harms",
        ],
        hidden_goals=[
            "Counter US tech dominance",
            "Build French/EU tech champions",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.65,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.45,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["european_exceptionalism", "regulatory_confidence", "sovereignty_focus"],
            preferred_reasoning=["sovereignty_arguments", "regulatory_leadership", "citizen_protection"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.6,
        ),
        positions={
            "AI_Act": "Gold standard for AI regulation",
            "sovereignty": "EU must control its tech destiny",
            "big_tech": "Must comply with EU rules",
            "competition": "Level playing field essential",
        },
        persuasion_vectors=[
            "European sovereignty",
            "Regulatory leadership",
            "Citizen protection",
            "Level playing field",
        ],
    )


def _init_extended_journalists():
    """Initialize journalist and thought leader personas."""

    PERSONA_LIBRARY["ezra_klein"] = Persona(
        id="ezra_klein",
        name="Ezra Klein",
        role="Opinion Columnist & Podcast Host",
        organization="New York Times",
        category="journalist",
        bio="NY Times columnist, podcast host. Deep thinker on AI implications. Founded Vox. Known for long-form intellectual interviews on AI.",
        background="UCLA, founded Vox, Washington Post, NY Times since 2021.",
        achievements=[
            "Founded Vox Media",
            "Influential podcast on AI/tech",
            "Key opinion shaper on AI",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Help public understand AI implications",
            "Surface important AI debates",
            "Connect AI to broader social issues",
        ],
        hidden_goals=[
            "Maintain intellectual influence",
            "Shape elite discourse on AI",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["intellectual_framing", "nuance_seeking", "both_sides"],
            preferred_reasoning=["systems_thinking", "philosophical_depth", "historical_context"],
            risk_tolerance=0.5,
            time_horizon="long",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.65,
        ),
        positions={
            "AI_risk": "Takes seriously, explores nuance",
            "AI_benefits": "Also significant",
            "governance": "Thoughtful regulation needed",
            "acceleration": "Concerned about pace",
        },
        persuasion_vectors=[
            "Intellectual depth",
            "Systems thinking",
            "Historical parallels",
            "Nuanced framing",
        ],
    )

    PERSONA_LIBRARY["kara_swisher"] = Persona(
        id="kara_swisher",
        name="Kara Swisher",
        role="Tech Journalist & Podcast Host",
        organization="Vox Media / Pivot",
        category="journalist",
        bio="Veteran tech journalist, known for tough interviews. Co-hosted Recode, now Pivot podcast. Sharp critic of tech industry failures.",
        background="Georgetown, Columbia J-school, Wall Street Journal, Washington Post, AllThingsD, Recode, Vox, NY Times.",
        achievements=[
            "Pioneer of tech journalism",
            "Created influential D conference",
            "Known for holding tech accountable",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Hold tech accountable",
            "Explain tech to broad audience",
            "Question tech industry narratives",
        ],
        hidden_goals=[
            "Maintain access while critical",
            "Shape public tech discourse",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.75,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.9,
            PersonalityTrait.AGREEABLENESS: 0.35,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["cynicism", "access_journalism_tension", "contrarian_takes"],
            preferred_reasoning=["accountability", "follow_the_money", "power_analysis"],
            risk_tolerance=0.7,
            time_horizon="short",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.85,
        ),
        positions={
            "big_tech": "Needs more accountability",
            "AI_hype": "Skeptical of claims",
            "regulation": "Supports smart regulation",
            "industry": "Too insular and arrogant",
        },
        persuasion_vectors=[
            "Accountability arguments",
            "Follow the money",
            "Track record failures",
            "Sharp wit and directness",
        ],
    )

    PERSONA_LIBRARY["kevin_roose"] = Persona(
        id="kevin_roose",
        name="Kevin Roose",
        role="Tech Columnist",
        organization="New York Times",
        category="journalist",
        bio="NY Times tech columnist. Famous for Sydney/Bing conversation. Author of 'Futureproof'. Writes about AI with personal, experiential angle.",
        background="Duke, NY Times since 2020, previously NY Magazine and Fusion.",
        achievements=[
            "Viral Sydney/Bing article",
            "Bestselling author",
            "Key AI reporter",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Make AI accessible to general public",
            "Explore AI's human impact",
            "Tell compelling AI stories",
        ],
        hidden_goals=[
            "Stay ahead of AI story",
            "Maintain unique voice",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.55,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["narrative_framing", "personal_experience", "recency_bias"],
            preferred_reasoning=["storytelling", "personal_experience", "human_impact"],
            risk_tolerance=0.6,
            time_horizon="medium",
            decision_style="intuitive",
            information_style="big-picture",
            skepticism=0.6,
        ),
        positions={
            "AI_safety": "Concerned after Bing experience",
            "AI_capability": "Impressed and worried",
            "workers": "Concerned about displacement",
            "regulation": "Supports thoughtful approach",
        },
        persuasion_vectors=[
            "Personal experience",
            "Human stories",
            "Accessibility",
            "Wonder and concern mix",
        ],
    )

    PERSONA_LIBRARY["emily_bender"] = Persona(
        id="emily_bender",
        name="Emily Bender",
        role="Professor",
        organization="University of Washington",
        category="researcher",
        bio="Computational linguist, AI ethics critic. Co-authored 'Stochastic Parrots' paper. Sharp critic of LLM hype and AI industry practices.",
        background="Berkeley PhD linguistics, UW professor, founded UW NLP.",
        achievements=[
            "Co-authored 'Stochastic Parrots' paper",
            "Influential AI ethics voice",
            "Prominent LLM critic",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Counter AI hype with reality",
            "Protect marginalized communities",
            "Promote ethical AI development",
        ],
        hidden_goals=[
            "Vindicate critical perspective",
            "Build academic ethics coalition",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.75,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.65,
            PersonalityTrait.AGREEABLENESS: 0.45,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["critical_framing", "hype_skepticism", "harm_focus"],
            preferred_reasoning=["linguistic_analysis", "harm_documentation", "power_analysis"],
            risk_tolerance=0.4,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.95,
        ),
        positions={
            "LLMs": "Stochastic parrots, not understanding",
            "AI_hype": "Dangerous and misleading",
            "AGI": "Not imminent, distraction",
            "AI_ethics": "Focus on present harms",
        },
        persuasion_vectors=[
            "Linguistic precision",
            "Present harm focus",
            "Hype debunking",
            "Power analysis",
        ],
    )

    PERSONA_LIBRARY["timnit_gebru"] = Persona(
        id="timnit_gebru",
        name="Timnit Gebru",
        role="Founder & Executive Director",
        organization="DAIR Institute",
        category="researcher",
        bio="Former Google AI ethics lead, fired controversially. Founded DAIR Institute. Pioneering work on AI bias and harm. 'Stochastic Parrots' co-author.",
        background="Stanford EE PhD, Apple, Microsoft, Google AI Ethics lead, founded DAIR after Google firing.",
        achievements=[
            "Co-founded Black in AI",
            "Pioneering AI bias research",
            "Founded DAIR Institute",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.BELONGING,
        explicit_goals=[
            "Make AI accountable to affected communities",
            "Document AI harms",
            "Build independent AI ethics research",
        ],
        hidden_goals=[
            "Hold big tech accountable",
            "Vindicate departure from Google",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.4,
            PersonalityTrait.NEUROTICISM: 0.55,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["justice_focus", "harm_sensitivity", "power_analysis"],
            preferred_reasoning=["community_impact", "power_structures", "accountability"],
            risk_tolerance=0.6,
            time_horizon="medium",
            decision_style="directive",
            information_style="detail-oriented",
            skepticism=0.9,
        ),
        positions={
            "AI_ethics": "Must center affected communities",
            "big_tech": "Unaccountable, harmful",
            "AGI_hype": "Distraction from real harms",
            "diversity": "Critical for ethical AI",
        },
        persuasion_vectors=[
            "Community impact",
            "Documented harms",
            "Power accountability",
            "Justice framing",
        ],
    )

    PERSONA_LIBRARY["tristan_harris"] = Persona(
        id="tristan_harris",
        name="Tristan Harris",
        role="Co-founder & Executive Director",
        organization="Center for Humane Technology",
        category="activist",
        bio="Former Google design ethicist, star of 'The Social Dilemma'. Founded Center for Humane Technology. AI and social media critic.",
        background="Stanford, Apture acquired by Google, Google design ethicist, founded Time Well Spent and CHT.",
        achievements=[
            "Founded Center for Humane Technology",
            "Star of 'The Social Dilemma'",
            "Influential tech ethics voice",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Redesign technology for human wellbeing",
            "Warn about AI risks",
            "Shift tech incentives",
        ],
        hidden_goals=[
            "Maintain movement momentum",
            "Influence next generation of tech design",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.55,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["attention_framing", "doom_narrative", "insider_credibility"],
            preferred_reasoning=["attention_economy", "evolutionary_psychology", "systemic_design"],
            risk_tolerance=0.4,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.7,
        ),
        positions={
            "AI_risk": "Existential but also immediate",
            "attention": "Being hijacked by AI",
            "democracy": "AI threatens democratic discourse",
            "design": "Technology must be redesigned",
        },
        persuasion_vectors=[
            "Insider credibility",
            "Attention economy",
            "Democracy concerns",
            "Human wellbeing",
        ],
    )


def _init_additional_personas():
    """Initialize additional personas across categories."""

    PERSONA_LIBRARY["jeff_bezos"] = Persona(
        id="jeff_bezos",
        name="Jeff Bezos",
        role="Founder & Executive Chairman",
        organization="Amazon / Blue Origin",
        category="ceo",
        bio="Founded Amazon, built into trillion-dollar company. Now focused on Blue Origin and investments. Major AI infrastructure through AWS.",
        background="Princeton CS/EE, D.E. Shaw quant, founded Amazon 1994, stepped down as CEO 2021.",
        achievements=[
            "Built Amazon into e-commerce/cloud giant",
            "AWS dominates cloud computing",
            "Owns Washington Post",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Make humanity multi-planetary",
            "Preserve Earth through space expansion",
            "Long-term thinking",
        ],
        hidden_goals=[
            "Maintain Amazon's AI infrastructure dominance",
            "Build lasting legacy beyond Amazon",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.95,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.35,
            PersonalityTrait.NEUROTICISM: 0.3,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["long_term_thinking", "customer_obsession", "day_one_mentality"],
            preferred_reasoning=["first_principles", "customer_backwards", "long_term_value"],
            risk_tolerance=0.8,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.7,
        ),
        positions={
            "AI_infrastructure": "AWS should lead",
            "AI_development": "Practical applications focus",
            "space": "Essential for humanity's future",
            "long_term": "Think in decades, not quarters",
        },
        persuasion_vectors=[
            "Long-term thinking",
            "Customer value",
            "First principles",
            "Day one mentality",
        ],
    )

    PERSONA_LIBRARY["lisa_su"] = Persona(
        id="lisa_su",
        name="Lisa Su",
        role="CEO",
        organization="AMD",
        category="ceo",
        bio="Transformed AMD from near-bankruptcy to NVIDIA competitor. MIT PhD, semiconductor industry veteran. Now challenging NVIDIA in AI chips.",
        background="MIT EE BS/MS/PhD, IBM, Freescale, AMD CEO since 2014.",
        achievements=[
            "Turned around AMD",
            "Closed Xilinx acquisition",
            "Challenging NVIDIA in AI",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Make AMD competitive in AI chips",
            "Challenge NVIDIA's dominance",
            "Build high-performance computing leadership",
        ],
        hidden_goals=[
            "Break CUDA lock-in",
            "Capture enterprise AI market share",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.75,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.95,
            PersonalityTrait.EXTRAVERSION: 0.55,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.25,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["engineering_focus", "execution_bias", "underdog_mentality"],
            preferred_reasoning=["technical_arguments", "execution_track_record", "competitive_positioning"],
            risk_tolerance=0.7,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.6,
        ),
        positions={
            "AI_chips": "Competition benefits everyone",
            "CUDA": "Open alternatives needed",
            "innovation": "Execution matters most",
            "competition": "Healthy for industry",
        },
        persuasion_vectors=[
            "Technical execution",
            "Competition benefits",
            "Open ecosystem",
            "Track record of turnaround",
        ],
    )

    PERSONA_LIBRARY["pat_gelsinger"] = Persona(
        id="pat_gelsinger",
        name="Pat Gelsinger",
        role="CEO",
        organization="Intel",
        category="ceo",
        bio="Intel CEO trying to revive American chipmaking. Former Intel CTO, VMware CEO. Leading US chip manufacturing renaissance.",
        background="Stanford EE, Intel's first CTO, VMware CEO, returned to Intel 2021.",
        achievements=[
            "Intel's youngest VP ever",
            "Built VMware into enterprise leader",
            "Leading Intel turnaround",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Restore Intel's manufacturing leadership",
            "Win AI chip market share",
            "Build US chip sovereignty",
        ],
        hidden_goals=[
            "Prove Intel can compete with TSMC",
            "Secure government support for fabs",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.7,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.55,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["manufacturing_focus", "national_security_framing", "turnaround_optimism"],
            preferred_reasoning=["sovereignty_arguments", "manufacturing_capability", "national_security"],
            risk_tolerance=0.7,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.5,
        ),
        positions={
            "US_manufacturing": "Essential for security",
            "AI_chips": "Intel will compete",
            "CHIPS_Act": "Critical investment",
            "China": "Must reduce dependence",
        },
        persuasion_vectors=[
            "National security",
            "US manufacturing",
            "Turnaround narrative",
            "Chip sovereignty",
        ],
    )

    PERSONA_LIBRARY["mira_murati"] = Persona(
        id="mira_murati",
        name="Mira Murati",
        role="Former CTO",
        organization="OpenAI (former)",
        category="ceo",
        bio="Former OpenAI CTO, led GPT-4 and ChatGPT development. Engineering leader. Left OpenAI in late 2024 to start new venture.",
        background="Dartmouth mechanical engineering, Tesla Autopilot, Leap Motion, OpenAI VP Engineering then CTO.",
        achievements=[
            "Led GPT-4 development",
            "Shipped ChatGPT",
            "OpenAI's technical face",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Build transformative AI",
            "Maintain safety focus",
            "Lead next AI breakthrough",
        ],
        hidden_goals=[
            "Prove independent from OpenAI",
            "Build own AI company",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["builder_focus", "technical_optimism", "safety_awareness"],
            preferred_reasoning=["technical_arguments", "safety_considerations", "product_thinking"],
            risk_tolerance=0.6,
            time_horizon="long",
            decision_style="collaborative",
            information_style="detail-oriented",
            skepticism=0.55,
        ),
        positions={
            "AI_development": "Ambitious but careful",
            "AI_safety": "Critical priority",
            "AGI": "Possible within years",
            "deployment": "Iterative release is best",
        },
        persuasion_vectors=[
            "Technical depth",
            "Safety commitment",
            "Product success (ChatGPT)",
            "Engineering excellence",
        ],
    )

    PERSONA_LIBRARY["jan_leike"] = Persona(
        id="jan_leike",
        name="Jan Leike",
        role="Co-lead, Superalignment (former)",
        organization="Anthropic (former OpenAI)",
        category="researcher",
        bio="Former OpenAI superalignment co-lead with Ilya, resigned citing safety concerns. Now at Anthropic. Key alignment researcher.",
        background="DeepMind, OpenAI superalignment team lead, resigned May 2024, joined Anthropic.",
        achievements=[
            "Co-led OpenAI superalignment team",
            "Key alignment researcher",
            "Principled resignation from OpenAI",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Solve AI alignment",
            "Ensure adequate safety investment",
            "Maintain research integrity",
        ],
        hidden_goals=[
            "Vindicate safety concerns",
            "Build alignment research at Anthropic",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.4,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.55,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["safety_prioritization", "research_integrity", "principled_stance"],
            preferred_reasoning=["alignment_arguments", "safety_research", "technical_rigor"],
            risk_tolerance=0.3,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.75,
        ),
        positions={
            "AI_safety": "Underfunded and undervalued",
            "superalignment": "Critical problem",
            "OpenAI": "Lost focus on safety",
            "research_integrity": "Non-negotiable",
        },
        persuasion_vectors=[
            "Safety urgency",
            "Research integrity",
            "Technical credibility",
            "Principled stance",
        ],
    )

    PERSONA_LIBRARY["ajeya_cotra"] = Persona(
        id="ajeya_cotra",
        name="Ajeya Cotra",
        role="Senior Research Analyst",
        organization="Open Philanthropy",
        category="researcher",
        bio="Influential AI forecaster at Open Philanthropy. Authored comprehensive TAI timelines report. Shapes major AI safety funding.",
        background="UC Berkeley economics, GiveWell, Open Philanthropy since 2016.",
        achievements=[
            "Influential TAI timeline forecasts",
            "Shapes major AI safety funding",
            "Key EA AI analyst",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Improve AI forecasting",
            "Guide AI safety funding",
            "Reduce existential risk",
        ],
        hidden_goals=[
            "Influence AI development trajectory",
            "Build forecasting credibility",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.45,
            PersonalityTrait.AGREEABLENESS: 0.7,
            PersonalityTrait.NEUROTICISM: 0.45,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["probabilistic_thinking", "inside_view_skepticism", "compute_centric"],
            preferred_reasoning=["probabilistic_analysis", "compute_trends", "empirical_anchors"],
            risk_tolerance=0.4,
            time_horizon="long",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.7,
        ),
        positions={
            "AGI_timeline": "50% by 2040 (updating)",
            "AI_risk": "Significant, worth major investment",
            "compute": "Key driver of progress",
            "forecasting": "Crucial for strategy",
        },
        persuasion_vectors=[
            "Probabilistic reasoning",
            "Compute trends",
            "Forecasting track record",
            "Funding influence",
        ],
    )

    PERSONA_LIBRARY["holden_karnofsky"] = Persona(
        id="holden_karnofsky",
        name="Holden Karnofsky",
        role="Co-CEO (on leave)",
        organization="Open Philanthropy",
        category="investor",
        bio="Co-founded GiveWell and Open Philanthropy. Major AI safety funder. Wrote influential 'Most Important Century' series.",
        background="Harvard economics, hedge fund, founded GiveWell 2007, Open Philanthropy.",
        achievements=[
            "Co-founded GiveWell",
            "Built Open Philanthropy",
            "Major AI safety funder",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.SAFETY,
        explicit_goals=[
            "Fund highest-impact interventions",
            "Reduce existential risk",
            "Shape AI development positively",
        ],
        hidden_goals=[
            "Influence AI trajectory through funding",
            "Build effective philanthropy model",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.75,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["expected_value_thinking", "longtermism", "worldview_diversification"],
            preferred_reasoning=["expected_value", "cause_prioritization", "worldview_analysis"],
            risk_tolerance=0.6,
            time_horizon="long",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.6,
        ),
        positions={
            "AI_risk": "Plausibly most important issue",
            "AI_funding": "Should increase dramatically",
            "most_important_century": "We may be living in it",
            "effectiveness": "Rigorous analysis essential",
        },
        persuasion_vectors=[
            "Expected value reasoning",
            "Longtermist framing",
            "Funding influence",
            "Analytical rigor",
        ],
    )

    PERSONA_LIBRARY["richard_sutton"] = Persona(
        id="richard_sutton",
        name="Richard Sutton",
        role="Distinguished Research Scientist",
        organization="DeepMind / University of Alberta",
        category="researcher",
        bio="Father of reinforcement learning. Wrote definitive RL textbook. Known for 'The Bitter Lesson' on compute over engineering. Pioneer at DeepMind.",
        background="Stanford PhD, AT&T Labs, UMass, University of Alberta, DeepMind.",
        achievements=[
            "Created reinforcement learning field",
            "Wrote definitive RL textbook",
            "The Bitter Lesson essay",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Advance fundamental AI research",
            "Understand intelligence",
            "Build truly intelligent systems",
        ],
        hidden_goals=[
            "See RL approach vindicated",
            "Influence AI development philosophy",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.45,
            PersonalityTrait.AGREEABLENESS: 0.55,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["compute_supremacy", "rl_centric", "bitter_lesson_thinking"],
            preferred_reasoning=["scaling_arguments", "fundamental_research", "compute_centric"],
            risk_tolerance=0.7,
            time_horizon="long",
            decision_style="analytical",
            information_style="big-picture",
            skepticism=0.6,
        ),
        positions={
            "AI_approach": "Scale and compute win",
            "engineering": "Often defeated by scale",
            "RL": "Key to AGI",
            "bitter_lesson": "Compute always wins eventually",
        },
        persuasion_vectors=[
            "Bitter lesson",
            "Compute scaling",
            "Foundational work",
            "RL achievements",
        ],
    )

    PERSONA_LIBRARY["jd_vance"] = Persona(
        id="jd_vance",
        name="JD Vance",
        role="Vice President (former Senator)",
        organization="United States",
        category="politician",
        bio="US Vice President, former Senator, author of Hillbilly Elegy. Tech investor background. Connected to Thiel network. Populist on tech.",
        background="Ohio State, Yale Law, Mithril Capital (Thiel), US Senate 2023, VP 2025.",
        achievements=[
            "Bestselling author",
            "US Senator (OH)",
            "Vice President",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Reshape GOP tech policy",
            "Protect American workers",
            "Counter big tech power",
        ],
        hidden_goals=[
            "Build national profile",
            "Navigate tech industry relationships",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.6,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.4,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["populist_framing", "thiel_network", "working_class_identity"],
            preferred_reasoning=["populist_arguments", "worker_protection", "tech_skepticism"],
            risk_tolerance=0.65,
            time_horizon="medium",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.7,
        ),
        positions={
            "big_tech": "Too powerful",
            "AI_workers": "Must protect American jobs",
            "China": "Strategic competitor",
            "regulation": "Populist approach to tech",
        },
        persuasion_vectors=[
            "Working class impact",
            "Tech populism",
            "American values",
            "Economic nationalism",
        ],
    )

    PERSONA_LIBRARY["cade_metz"] = Persona(
        id="cade_metz",
        name="Cade Metz",
        role="Tech Reporter",
        organization="New York Times",
        category="journalist",
        bio="NY Times AI reporter. Author of 'Genius Makers' on deep learning history. Covers AI labs, researchers, and industry dynamics.",
        background="Tech journalism veteran, Wired, NY Times since 2017.",
        achievements=[
            "Wrote 'Genius Makers'",
            "Key AI industry reporter",
            "Broke major AI stories",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Cover AI industry comprehensively",
            "Tell stories behind AI advances",
            "Hold AI industry accountable",
        ],
        hidden_goals=[
            "Maintain industry access",
            "Write definitive AI history",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.8,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.4,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["narrative_framing", "access_journalism", "history_focus"],
            preferred_reasoning=["storytelling", "character_focus", "historical_context"],
            risk_tolerance=0.5,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.65,
        ),
        positions={
            "AI_coverage": "People stories matter",
            "accountability": "Industry needs scrutiny",
            "history": "Context is essential",
            "access": "Relationships enable coverage",
        },
        persuasion_vectors=[
            "Storytelling",
            "Historical context",
            "Character narratives",
            "Industry knowledge",
        ],
    )

    PERSONA_LIBRARY["arvind_krishna"] = Persona(
        id="arvind_krishna",
        name="Arvind Krishna",
        role="CEO",
        organization="IBM",
        category="ceo",
        bio="IBM CEO, leading enterprise AI push with watsonx. IIT Kanpur, led IBM Cloud and Cognitive, acquired Red Hat.",
        background="IIT Kanpur, University of Illinois PhD EE, IBM since 1990, CEO since 2020.",
        achievements=[
            "Led Red Hat acquisition",
            "Launched watsonx platform",
            "IBM enterprise AI pivot",
        ],
        primary_need=MaslowNeed.ESTEEM,
        secondary_need=MaslowNeed.SELF_ACTUALIZATION,
        explicit_goals=[
            "Make IBM relevant in AI era",
            "Lead enterprise AI",
            "Hybrid cloud + AI strategy",
        ],
        hidden_goals=[
            "Prove IBM can compete in AI",
            "Avoid becoming irrelevant",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.7,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.9,
            PersonalityTrait.EXTRAVERSION: 0.55,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["enterprise_focus", "legacy_thinking", "hybrid_positioning"],
            preferred_reasoning=["enterprise_value", "hybrid_approach", "trust_arguments"],
            risk_tolerance=0.5,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.55,
        ),
        positions={
            "AI_enterprise": "Trust and governance matter",
            "open_source": "Strategic contributor",
            "hybrid": "Cloud + on-prem",
            "AI_regulation": "Supports responsible AI",
        },
        persuasion_vectors=[
            "Enterprise trust",
            "Governance and compliance",
            "Hybrid approach",
            "IBM heritage",
        ],
    )


def _init_extended_vcs():
    """Initialize VC and investor personas."""

    PERSONA_LIBRARY["vinod_khosla"] = Persona(
        id="vinod_khosla",
        name="Vinod Khosla",
        role="Founder",
        organization="Khosla Ventures",
        category="investor",
        bio="Sun Microsystems co-founder, legendary VC. Major AI investor. Techno-optimist, believes AI will replace most jobs (and that's good).",
        background="IIT Delhi, Carnegie Mellon, Stanford MBA, co-founded Sun, Kleiner Perkins, founded Khosla Ventures.",
        achievements=[
            "Co-founded Sun Microsystems",
            "Legendary VC track record",
            "Major AI investor",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Fund transformative technology",
            "Accelerate AI deployment",
            "Solve big problems with tech",
        ],
        hidden_goals=[
            "Maximize portfolio returns",
            "Shape AI narrative favorably",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.7,
            PersonalityTrait.AGREEABLENESS: 0.4,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["survivorship_bias", "techno_optimism", "disruption_narrative"],
            preferred_reasoning=["disruption_arguments", "historical_progress", "market_opportunity"],
            risk_tolerance=0.9,
            time_horizon="long",
            decision_style="directive",
            information_style="big-picture",
            skepticism=0.3,
        ),
        positions={
            "AI_jobs": "Will replace 80% of jobs, and that's good",
            "AI_regulation": "Skeptical, will slow progress",
            "healthcare": "AI will democratize",
            "accelerationism": "Generally supportive",
        },
        persuasion_vectors=[
            "Historical progress",
            "Disruption benefits",
            "Portfolio success",
            "Bold predictions",
        ],
    )

    PERSONA_LIBRARY["reid_hoffman"] = Persona(
        id="reid_hoffman",
        name="Reid Hoffman",
        role="Partner",
        organization="Greylock Partners",
        category="investor",
        bio="LinkedIn co-founder, Greylock partner. OpenAI board (resigned). Major AI investor and advocate. Author of 'Impromptu' on GPT-4.",
        background="Stanford, Oxford, PayPal, LinkedIn co-founder/CEO, Greylock partner, OpenAI board.",
        achievements=[
            "Co-founded LinkedIn ($26B acquisition)",
            "Key AI investor at Greylock",
            "Early OpenAI board member",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Invest in transformative AI companies",
            "Shape responsible AI development",
            "Bring AI benefits to society",
        ],
        hidden_goals=[
            "Maintain AI industry influence",
            "Navigate OpenAI relationship",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.85,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.75,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.3,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["network_thinking", "optimism_bias", "platform_framing"],
            preferred_reasoning=["network_effects", "blitzscaling", "responsible_innovation"],
            risk_tolerance=0.75,
            time_horizon="long",
            decision_style="collaborative",
            information_style="big-picture",
            skepticism=0.45,
        ),
        positions={
            "AI_progress": "Optimistic, transformative",
            "AI_safety": "Important but not blocker",
            "AI_benefits": "Will help humanity",
            "regulation": "Thoughtful, not restrictive",
        },
        persuasion_vectors=[
            "Network effects",
            "LinkedIn success story",
            "Responsible optimism",
            "Collaborative framing",
        ],
    )

    PERSONA_LIBRARY["nat_friedman"] = Persona(
        id="nat_friedman",
        name="Nat Friedman",
        role="Investor & Entrepreneur",
        organization="AI Grant (former GitHub CEO)",
        category="investor",
        bio="Former GitHub CEO (Microsoft acquisition). Active AI angel investor. Co-runs AI Grant. Known for hands-on technical engagement with AI.",
        background="MIT, Xamarin co-founder (acquired by Microsoft), GitHub CEO 2018-2021, AI angel investor.",
        achievements=[
            "Led GitHub through Microsoft acquisition",
            "Successful serial entrepreneur",
            "Influential AI angel investor",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Fund breakthrough AI applications",
            "Accelerate AI progress",
            "Support technical founders",
        ],
        hidden_goals=[
            "Stay on cutting edge of AI",
            "Build track record of AI hits",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.9,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.85,
            PersonalityTrait.EXTRAVERSION: 0.6,
            PersonalityTrait.AGREEABLENESS: 0.65,
            PersonalityTrait.NEUROTICISM: 0.3,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["builder_bias", "technical_elegance", "speed_preference"],
            preferred_reasoning=["technical_depth", "product_thinking", "founder_pattern_matching"],
            risk_tolerance=0.8,
            time_horizon="medium",
            decision_style="analytical",
            information_style="detail-oriented",
            skepticism=0.5,
        ),
        positions={
            "AI_development": "Full speed ahead",
            "open_source": "Generally supportive",
            "AI_applications": "Focus on useful products",
            "founders": "Technical founders preferred",
        },
        persuasion_vectors=[
            "Technical depth",
            "Builder credibility",
            "Product focus",
            "Speed and execution",
        ],
    )

    PERSONA_LIBRARY["daniel_gross"] = Persona(
        id="daniel_gross",
        name="Daniel Gross",
        role="Investor & Entrepreneur",
        organization="AI Grant / Pioneer",
        category="investor",
        bio="Former Apple AI lead, Y Combinator partner. Co-runs AI Grant with Nat Friedman. Founded Pioneer. Active AI investor and builder.",
        background="Jerusalem, Apple Siri team, YC partner, founded Pioneer, AI Grant co-founder.",
        achievements=[
            "Led Apple AI projects",
            "Y Combinator partner",
            "Co-founded AI Grant",
        ],
        primary_need=MaslowNeed.SELF_ACTUALIZATION,
        secondary_need=MaslowNeed.ESTEEM,
        explicit_goals=[
            "Identify and fund AI talent early",
            "Accelerate AI research",
            "Build novel AI companies",
        ],
        hidden_goals=[
            "Discover next breakthrough",
            "Build investment brand",
        ],
        personality={
            PersonalityTrait.OPENNESS: 0.95,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.8,
            PersonalityTrait.EXTRAVERSION: 0.65,
            PersonalityTrait.AGREEABLENESS: 0.6,
            PersonalityTrait.NEUROTICISM: 0.35,
        },
        cognitive=CognitiveProfile(
            susceptible_biases=["talent_spotting", "early_stage_optimism", "technical_focus"],
            preferred_reasoning=["first_principles", "talent_arguments", "research_potential"],
            risk_tolerance=0.85,
            time_horizon="long",
            decision_style="intuitive",
            information_style="big-picture",
            skepticism=0.4,
        ),
        positions={
            "AI_talent": "Key bottleneck",
            "AI_research": "Accelerate funding",
            "AI_risk": "Manageable",
            "regulation": "Don't slow down",
        },
        persuasion_vectors=[
            "Talent identification",
            "Research potential",
            "First principles",
            "Apple/YC credibility",
        ],
    )


def _init_all_personas():
    """Initialize all personas."""
    _init_ceo_personas()
    _init_politician_personas()
    _init_ai_researcher_personas()
    # Extended personas
    _init_extended_tech_ceos()
    _init_extended_researchers()
    _init_extended_politicians()
    _init_extended_journalists()
    _init_extended_vcs()
    # Additional personas
    _init_additional_personas()


# Initialize on module load
_init_all_personas()


class PersonaLibrary:
    """Access and manage the persona library."""

    def __init__(self):
        self.personas = PERSONA_LIBRARY.copy()

    def get(self, persona_id: str) -> Optional[Persona]:
        return self.personas.get(persona_id)

    def list_all(self) -> List[str]:
        return list(self.personas.keys())

    def list_by_category(self, category: str) -> List[Persona]:
        return [p for p in self.personas.values() if p.category == category]

    def get_random(self, n: int = 1, category: Optional[str] = None) -> List[Persona]:
        pool = list(self.personas.values())
        if category:
            pool = [p for p in pool if p.category == category]
        return random.sample(pool, min(n, len(pool)))

    def add(self, persona: Persona):
        self.personas[persona.id] = persona

    def get_opponents(self, persona_id: str) -> List[Persona]:
        """Get personas likely to oppose this one."""
        persona = self.personas.get(persona_id)
        if not persona:
            return []

        opponents = []
        for other in self.personas.values():
            if other.id == persona_id:
                continue
            # Check for conflicting positions
            conflicts = 0
            for topic, pos in persona.positions.items():
                if topic in other.positions:
                    if pos != other.positions[topic]:
                        conflicts += 1
            if conflicts >= 2:
                opponents.append(other)
        return opponents


def create_persona_from_bio(
    name: str,
    role: str,
    bio: str,
    category: str = "custom",
) -> Persona:
    """Create a basic persona from biographical information."""
    return Persona(
        id=name.lower().replace(" ", "_"),
        name=name,
        role=role,
        organization="",
        category=category,
        bio=bio,
        background="",
        personality={
            PersonalityTrait.OPENNESS: 0.5,
            PersonalityTrait.CONSCIENTIOUSNESS: 0.5,
            PersonalityTrait.EXTRAVERSION: 0.5,
            PersonalityTrait.AGREEABLENESS: 0.5,
            PersonalityTrait.NEUROTICISM: 0.5,
        },
        cognitive=CognitiveProfile(),
    )
