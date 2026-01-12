"""
Advanced Psychological Modeling for Multi-Agent Simulations

This module provides sophisticated psychological modeling including:
- Theory of Mind (modeling others' beliefs, desires, intentions)
- Cognitive Biases (systematic deviations from rationality)
- Emotional Contagion (how emotions spread between agents)
- Social Influence Dynamics
- Personality Systems (Big Five, MBTI-inspired, custom traits)
- Relationship Dynamics
- Trust and Reputation Systems
- Group Dynamics and Social Identity

Based on research in:
- Social Psychology
- Cognitive Science
- Behavioral Economics
- Game Theory
- Complex Systems
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


# ============================================================================
# PERSONALITY SYSTEMS
# ============================================================================

class BigFiveTrait(Enum):
    """Big Five / OCEAN personality traits"""
    OPENNESS = "openness"  # Curiosity, creativity, openness to experience
    CONSCIENTIOUSNESS = "conscientiousness"  # Organization, dependability
    EXTRAVERSION = "extraversion"  # Sociability, assertiveness
    AGREEABLENESS = "agreeableness"  # Cooperation, trust, altruism
    NEUROTICISM = "neuroticism"  # Emotional instability, anxiety


@dataclass
class PersonalityProfile:
    """Complete personality profile for an agent"""
    # Big Five traits (0.0 to 1.0)
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    # Additional psychological traits
    need_for_cognition: float = 0.5  # Enjoyment of thinking
    social_dominance: float = 0.5  # Preference for hierarchy
    empathy: float = 0.5
    impulsivity: float = 0.5
    risk_tolerance: float = 0.5
    need_for_closure: float = 0.5  # Desire for definite answers
    authoritarianism: float = 0.5
    narcissism: float = 0.3  # Healthy level ~0.3
    machiavellianism: float = 0.3
    psychopathy: float = 0.1  # Very low in healthy individuals

    # Derived traits (computed)
    @property
    def emotional_stability(self) -> float:
        return 1.0 - self.neuroticism

    @property
    def social_sensitivity(self) -> float:
        return (self.agreeableness + self.empathy) / 2

    @property
    def cognitive_flexibility(self) -> float:
        return (self.openness + (1 - self.need_for_closure)) / 2

    @property
    def dark_triad_score(self) -> float:
        return (self.narcissism + self.machiavellianism + self.psychopathy) / 3

    def get_trait(self, trait: BigFiveTrait) -> float:
        return getattr(self, trait.value)

    def modify_trait(self, trait: str, delta: float):
        """Modify a trait with bounds checking"""
        current = getattr(self, trait, 0.5)
        setattr(self, trait, max(0.0, min(1.0, current + delta)))

    def to_dict(self) -> Dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "need_for_cognition": self.need_for_cognition,
            "empathy": self.empathy,
            "risk_tolerance": self.risk_tolerance,
            "dark_triad": self.dark_triad_score
        }


def generate_random_personality(archetype: str = None) -> PersonalityProfile:
    """Generate a random personality, optionally based on an archetype"""
    archetypes = {
        "leader": {"extraversion": 0.8, "conscientiousness": 0.7, "social_dominance": 0.7},
        "intellectual": {"openness": 0.9, "need_for_cognition": 0.9, "introversion": 0.3},
        "empath": {"agreeableness": 0.9, "empathy": 0.9, "neuroticism": 0.5},
        "rebel": {"openness": 0.7, "agreeableness": 0.3, "authoritarianism": 0.2},
        "conformist": {"conscientiousness": 0.8, "need_for_closure": 0.8, "authoritarianism": 0.7},
        "manipulator": {"machiavellianism": 0.8, "empathy": 0.4, "narcissism": 0.6},
        "anxious": {"neuroticism": 0.8, "conscientiousness": 0.6, "risk_tolerance": 0.2},
        "adventurer": {"openness": 0.8, "risk_tolerance": 0.8, "impulsivity": 0.6},
    }

    base = PersonalityProfile()

    # Add random variation
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        base.modify_trait(trait, random.gauss(0, 0.15))

    # Apply archetype if specified
    if archetype and archetype in archetypes:
        for trait, value in archetypes[archetype].items():
            if trait == "introversion":
                base.extraversion = 1 - value
            else:
                setattr(base, trait, value + random.gauss(0, 0.1))

    return base


# ============================================================================
# COGNITIVE BIASES
# ============================================================================

class BiasType(Enum):
    """Types of cognitive biases"""
    # Decision-making biases
    CONFIRMATION_BIAS = "confirmation_bias"
    ANCHORING = "anchoring"
    AVAILABILITY_HEURISTIC = "availability"
    REPRESENTATIVENESS = "representativeness"
    LOSS_AVERSION = "loss_aversion"
    SUNK_COST = "sunk_cost"
    OVERCONFIDENCE = "overconfidence"
    HINDSIGHT = "hindsight"

    # Social biases
    INGROUP_BIAS = "ingroup_bias"
    OUTGROUP_HOMOGENEITY = "outgroup_homogeneity"
    HALO_EFFECT = "halo_effect"
    AUTHORITY_BIAS = "authority_bias"
    BANDWAGON = "bandwagon"
    FUNDAMENTAL_ATTRIBUTION = "fundamental_attribution"

    # Self-serving biases
    SELF_SERVING = "self_serving"
    OPTIMISM_BIAS = "optimism_bias"
    SPOTLIGHT_EFFECT = "spotlight_effect"

    # Memory biases
    RECENCY = "recency"
    PRIMACY = "primacy"
    PEAK_END = "peak_end"


@dataclass
class BiasProfile:
    """Profile of cognitive biases for an agent"""
    bias_strengths: Dict[BiasType, float] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize all biases with default strengths
        for bias in BiasType:
            if bias not in self.bias_strengths:
                self.bias_strengths[bias] = random.uniform(0.3, 0.7)

    def get_bias_strength(self, bias: BiasType) -> float:
        return self.bias_strengths.get(bias, 0.5)

    def apply_confirmation_bias(self, evidence: Dict[str, Any],
                                 prior_belief: float) -> float:
        """Apply confirmation bias to evidence evaluation"""
        strength = self.get_bias_strength(BiasType.CONFIRMATION_BIAS)
        evidence_strength = evidence.get("strength", 0.5)

        # Bias towards confirming existing belief
        if (evidence_strength > 0.5 and prior_belief > 0.5) or \
           (evidence_strength < 0.5 and prior_belief < 0.5):
            # Confirming evidence - amplify
            return evidence_strength + (strength * 0.3 * (1 - abs(evidence_strength - 0.5)))
        else:
            # Disconfirming evidence - discount
            return evidence_strength - (strength * 0.3 * abs(evidence_strength - 0.5))

    def apply_anchoring(self, initial_value: float, new_info: float) -> float:
        """Apply anchoring bias - insufficient adjustment from initial"""
        strength = self.get_bias_strength(BiasType.ANCHORING)
        adjustment = new_info - initial_value
        actual_adjustment = adjustment * (1 - strength * 0.5)
        return initial_value + actual_adjustment

    def apply_availability(self, event_probability: float,
                          recent_occurrences: int) -> float:
        """Apply availability heuristic"""
        strength = self.get_bias_strength(BiasType.AVAILABILITY_HEURISTIC)
        recency_boost = min(0.4, recent_occurrences * 0.1 * strength)
        return min(1.0, event_probability + recency_boost)

    def apply_loss_aversion(self, gain: float, loss: float) -> float:
        """Apply loss aversion - losses loom larger than gains"""
        strength = self.get_bias_strength(BiasType.LOSS_AVERSION)
        loss_weight = 1 + strength * 1.5  # Losses weighted 1.5-2.5x more
        return gain - (loss * loss_weight)


# ============================================================================
# THEORY OF MIND
# ============================================================================

@dataclass
class MentalStateModel:
    """Model of another agent's mental state"""
    agent_id: str
    beliefs: Dict[str, float] = field(default_factory=dict)  # What we think they believe
    desires: Dict[str, float] = field(default_factory=dict)  # What we think they want
    intentions: List[str] = field(default_factory=list)  # What we think they'll do
    emotions: Dict[str, float] = field(default_factory=dict)  # Their emotional state
    knowledge: Set[str] = field(default_factory=set)  # What we think they know
    trust_in_us: float = 0.5  # How much we think they trust us
    confidence: float = 0.5  # How confident we are in this model
    last_updated: datetime = field(default_factory=datetime.now)


class TheoryOfMind:
    """
    Theory of Mind system - modeling other agents' mental states.
    Enables social reasoning, deception detection, and cooperation.
    """

    def __init__(self, owner_id: str, personality: PersonalityProfile = None):
        self.owner_id = owner_id
        self.personality = personality or PersonalityProfile()
        self.models: Dict[str, MentalStateModel] = {}
        self.interaction_history: Dict[str, List[Dict]] = defaultdict(list)

        # Calibration based on personality
        self.empathy_factor = self.personality.empathy
        self.projection_tendency = 1 - self.personality.openness  # Project own states
        self.suspicion_level = self.personality.neuroticism * 0.5

    def get_or_create_model(self, agent_id: str) -> MentalStateModel:
        """Get existing model or create new one"""
        if agent_id not in self.models:
            self.models[agent_id] = MentalStateModel(agent_id=agent_id)
        return self.models[agent_id]

    def update_from_observation(self, agent_id: str, observation: Dict[str, Any]):
        """Update mental model based on observation"""
        model = self.get_or_create_model(agent_id)

        # Update based on observed behavior
        if "statement" in observation:
            self._infer_from_statement(model, observation["statement"])

        if "action" in observation:
            self._infer_from_action(model, observation["action"])

        if "emotion_display" in observation:
            self._update_emotional_model(model, observation["emotion_display"])

        model.last_updated = datetime.now()
        model.confidence = min(0.95, model.confidence + 0.05)

    def _infer_from_statement(self, model: MentalStateModel, statement: Dict):
        """Infer beliefs and knowledge from statements"""
        topic = statement.get("topic", "unknown")
        position = statement.get("position", 0.5)
        certainty = statement.get("certainty", 0.5)

        # Update beliefs
        model.beliefs[topic] = position

        # Infer knowledge
        if certainty > 0.7:
            model.knowledge.add(topic)

        # Detect potential deception (high suspicion personality)
        if self.suspicion_level > 0.5:
            # Look for inconsistencies
            if topic in model.beliefs:
                prior = model.beliefs[topic]
                if abs(prior - position) > 0.4:
                    # Major shift - suspicious
                    model.confidence *= 0.8

    def _infer_from_action(self, model: MentalStateModel, action: Dict):
        """Infer intentions from actions"""
        action_type = action.get("type", "unknown")
        target = action.get("target")

        # Infer intentions from patterns
        if action_type == "help":
            model.desires["cooperation"] = model.desires.get("cooperation", 0.5) + 0.1
        elif action_type == "compete":
            model.desires["dominance"] = model.desires.get("dominance", 0.5) + 0.1
        elif action_type == "share":
            model.trust_in_us = min(1.0, model.trust_in_us + 0.1)

    def _update_emotional_model(self, model: MentalStateModel, emotion: Dict):
        """Update model of agent's emotional state"""
        for emotion_name, intensity in emotion.items():
            model.emotions[emotion_name] = intensity

    def predict_response(self, agent_id: str, stimulus: Dict) -> Dict[str, Any]:
        """Predict how an agent will respond to a stimulus"""
        model = self.get_or_create_model(agent_id)

        prediction = {
            "agent_id": agent_id,
            "stimulus": stimulus,
            "predicted_emotion": {},
            "predicted_action": None,
            "confidence": model.confidence * 0.8
        }

        stimulus_type = stimulus.get("type", "neutral")
        stimulus_valence = stimulus.get("valence", 0.0)

        # Predict emotional response
        if stimulus_valence > 0.3:
            prediction["predicted_emotion"]["joy"] = 0.5 + model.emotions.get("baseline_positivity", 0)
        elif stimulus_valence < -0.3:
            prediction["predicted_emotion"]["distress"] = 0.5 + model.emotions.get("sensitivity", 0)

        # Predict action based on desires and beliefs
        if "conflict" in stimulus_type:
            if model.desires.get("dominance", 0) > 0.6:
                prediction["predicted_action"] = "escalate"
            elif model.desires.get("cooperation", 0) > 0.6:
                prediction["predicted_action"] = "de-escalate"
            else:
                prediction["predicted_action"] = "observe"

        return prediction

    def infer_belief(self, agent_id: str, topic: str) -> Tuple[float, float]:
        """Infer what an agent believes about a topic. Returns (belief, confidence)"""
        model = self.get_or_create_model(agent_id)

        if topic in model.beliefs:
            return model.beliefs[topic], model.confidence

        # Use projection if we don't know
        # (Assume others think like us, modulated by openness)
        projected = 0.5  # Default neutral
        return projected, model.confidence * 0.5

    def estimate_trust(self, agent_id: str) -> float:
        """Estimate how much an agent trusts us"""
        model = self.get_or_create_model(agent_id)
        return model.trust_in_us

    def detect_deception(self, agent_id: str, claimed_state: Dict) -> float:
        """Attempt to detect if agent is being deceptive. Returns probability."""
        model = self.get_or_create_model(agent_id)

        inconsistencies = 0
        checks = 0

        # Check for belief inconsistencies
        for topic, claimed_belief in claimed_state.get("beliefs", {}).items():
            if topic in model.beliefs:
                checks += 1
                if abs(claimed_belief - model.beliefs[topic]) > 0.4:
                    inconsistencies += 1

        # Check for emotion-behavior mismatch
        if "emotions" in claimed_state:
            for emotion, intensity in claimed_state["emotions"].items():
                if emotion in model.emotions:
                    checks += 1
                    if abs(intensity - model.emotions[emotion]) > 0.5:
                        inconsistencies += 1

        if checks == 0:
            return 0.5  # Unknown

        base_deception_prob = inconsistencies / checks

        # Modulate by suspicion level
        adjusted = base_deception_prob * (1 + self.suspicion_level * 0.5)

        return min(0.95, adjusted)


# ============================================================================
# EMOTIONAL CONTAGION
# ============================================================================

class EmotionType(Enum):
    """Basic emotions (Plutchik's wheel)"""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionalState:
    """Current emotional state of an agent"""
    emotions: Dict[EmotionType, float] = field(default_factory=dict)
    mood: float = 0.0  # Overall valence (-1 to 1)
    arousal: float = 0.5  # Activation level (0 to 1)
    stability: float = 0.5  # How stable emotions are

    def __post_init__(self):
        for emotion in EmotionType:
            if emotion not in self.emotions:
                self.emotions[emotion] = 0.0

    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """Get the strongest current emotion"""
        return max(self.emotions.items(), key=lambda x: x[1])

    def update_mood(self):
        """Update overall mood from emotions"""
        positive = self.emotions[EmotionType.JOY] + self.emotions[EmotionType.TRUST] + \
                   self.emotions[EmotionType.ANTICIPATION]
        negative = self.emotions[EmotionType.FEAR] + self.emotions[EmotionType.SADNESS] + \
                   self.emotions[EmotionType.ANGER] + self.emotions[EmotionType.DISGUST]
        self.mood = (positive - negative) / 7

    def decay(self, rate: float = 0.1):
        """Decay emotions towards baseline"""
        for emotion in self.emotions:
            self.emotions[emotion] *= (1 - rate)
        self.arousal *= (1 - rate * 0.5)


class EmotionalContagion:
    """
    Models how emotions spread between agents.
    Based on emotional contagion research by Hatfield et al.
    """

    def __init__(self):
        self.susceptibility: Dict[str, float] = {}  # How susceptible each agent is
        self.expressiveness: Dict[str, float] = {}  # How expressive each agent is
        self.history: List[Dict] = []

    def set_agent_properties(self, agent_id: str, susceptibility: float = 0.5,
                            expressiveness: float = 0.5):
        """Set contagion properties for an agent"""
        self.susceptibility[agent_id] = susceptibility
        self.expressiveness[agent_id] = expressiveness

    def calculate_contagion(self, source_id: str, source_state: EmotionalState,
                           target_id: str, target_state: EmotionalState,
                           interaction_intensity: float = 0.5) -> Dict[EmotionType, float]:
        """
        Calculate emotional contagion from source to target.
        Returns emotion deltas for target.
        """
        source_expr = self.expressiveness.get(source_id, 0.5)
        target_susc = self.susceptibility.get(target_id, 0.5)

        # Contagion strength
        strength = source_expr * target_susc * interaction_intensity

        deltas = {}
        for emotion in EmotionType:
            source_intensity = source_state.emotions.get(emotion, 0)
            target_intensity = target_state.emotions.get(emotion, 0)

            # Contagion pulls target toward source
            diff = source_intensity - target_intensity
            deltas[emotion] = diff * strength * 0.3

        self.history.append({
            "source": source_id,
            "target": target_id,
            "strength": strength,
            "timestamp": datetime.now()
        })

        return deltas

    def apply_group_contagion(self, states: Dict[str, EmotionalState],
                              proximity: Dict[Tuple[str, str], float]) -> Dict[str, Dict[EmotionType, float]]:
        """
        Calculate contagion effects across a group.
        Proximity is a dict of (agent1, agent2) -> proximity score.
        """
        all_deltas: Dict[str, Dict[EmotionType, float]] = defaultdict(lambda: defaultdict(float))

        for (a1, a2), prox in proximity.items():
            if a1 in states and a2 in states:
                # Bidirectional contagion
                deltas1 = self.calculate_contagion(a2, states[a2], a1, states[a1], prox)
                deltas2 = self.calculate_contagion(a1, states[a1], a2, states[a2], prox)

                for emotion, delta in deltas1.items():
                    all_deltas[a1][emotion] += delta
                for emotion, delta in deltas2.items():
                    all_deltas[a2][emotion] += delta

        return dict(all_deltas)


# ============================================================================
# SOCIAL INFLUENCE
# ============================================================================

class InfluenceType(Enum):
    """Types of social influence"""
    INFORMATIONAL = "informational"  # We believe they have info
    NORMATIVE = "normative"  # We want to fit in
    IDENTIFICATION = "identification"  # We want to be like them
    COMPLIANCE = "compliance"  # They have power over us
    INTERNALIZATION = "internalization"  # We've adopted their view


@dataclass
class InfluenceAttempt:
    """Record of an influence attempt"""
    source_id: str
    target_id: str
    influence_type: InfluenceType
    topic: str
    strength: float
    success: bool
    resistance_level: float
    timestamp: datetime = field(default_factory=datetime.now)


class SocialInfluenceModel:
    """
    Models how agents influence each other's beliefs and behaviors.
    """

    def __init__(self):
        self.influence_history: List[InfluenceAttempt] = []
        self.credibility: Dict[str, float] = {}  # Source credibility
        self.group_norms: Dict[str, Dict[str, float]] = {}  # Group -> topic -> norm

    def calculate_influence_potential(self, source: PersonalityProfile,
                                      target: PersonalityProfile,
                                      relationship_strength: float,
                                      source_credibility: float) -> float:
        """Calculate how much influence source could have on target"""
        # Target susceptibility
        target_openness = target.openness
        target_need_for_closure = target.need_for_closure
        target_authoritarianism = target.authoritarianism

        # Source factors
        source_dominance = source.social_dominance
        source_extraversion = source.extraversion

        # Calculate influence potential
        susceptibility = (target_openness * 0.3 +
                         (1 - target_need_for_closure) * 0.2 +
                         target_authoritarianism * 0.2)  # Authoritarians influenced by authority

        persuasiveness = (source_dominance * 0.3 +
                         source_extraversion * 0.2 +
                         source_credibility * 0.3)

        return susceptibility * persuasiveness * relationship_strength

    def attempt_influence(self, source_id: str, target_id: str,
                         source_profile: PersonalityProfile,
                         target_profile: PersonalityProfile,
                         topic: str,
                         influence_type: InfluenceType,
                         argument_strength: float = 0.5) -> InfluenceAttempt:
        """Attempt to influence a target agent"""
        credibility = self.credibility.get(source_id, 0.5)
        relationship = 0.5  # Default, would come from relationship system

        potential = self.calculate_influence_potential(
            source_profile, target_profile, relationship, credibility
        )

        # Calculate resistance
        resistance = 1 - target_profile.agreeableness
        if influence_type == InfluenceType.NORMATIVE:
            # Lower resistance if high need for social acceptance
            resistance *= (1 - target_profile.extraversion * 0.3)

        # Determine success
        influence_strength = potential * argument_strength
        success = influence_strength > resistance

        attempt = InfluenceAttempt(
            source_id=source_id,
            target_id=target_id,
            influence_type=influence_type,
            topic=topic,
            strength=influence_strength,
            success=success,
            resistance_level=resistance
        )

        self.influence_history.append(attempt)

        # Update credibility based on outcome
        if success:
            self.credibility[source_id] = min(1.0, credibility + 0.05)
        else:
            self.credibility[source_id] = max(0.0, credibility - 0.02)

        return attempt


# ============================================================================
# TRUST AND REPUTATION
# ============================================================================

@dataclass
class TrustRecord:
    """Record of trust between two agents"""
    trustor_id: str
    trustee_id: str
    cognitive_trust: float = 0.5  # Based on competence
    affective_trust: float = 0.5  # Based on emotional bond
    behavioral_trust: float = 0.5  # Based on past actions
    betrayal_history: List[datetime] = field(default_factory=list)
    last_interaction: datetime = field(default_factory=datetime.now)

    @property
    def overall_trust(self) -> float:
        base = (self.cognitive_trust * 0.3 +
                self.affective_trust * 0.3 +
                self.behavioral_trust * 0.4)
        # Betrayals reduce trust significantly
        betrayal_penalty = min(0.5, len(self.betrayal_history) * 0.15)
        return max(0.0, base - betrayal_penalty)


class TrustNetwork:
    """
    Network of trust relationships between agents.
    """

    def __init__(self):
        self.trust_records: Dict[Tuple[str, str], TrustRecord] = {}
        self.reputation: Dict[str, float] = {}  # Global reputation scores
        self.gossip_network: Dict[str, Set[str]] = defaultdict(set)

    def get_trust(self, trustor: str, trustee: str) -> TrustRecord:
        """Get trust record, creating if necessary"""
        key = (trustor, trustee)
        if key not in self.trust_records:
            self.trust_records[key] = TrustRecord(trustor_id=trustor, trustee_id=trustee)
        return self.trust_records[key]

    def update_trust(self, trustor: str, trustee: str, event: Dict[str, Any]):
        """Update trust based on an event"""
        record = self.get_trust(trustor, trustee)

        event_type = event.get("type", "neutral")
        magnitude = event.get("magnitude", 0.1)

        if event_type == "positive_outcome":
            record.behavioral_trust = min(1.0, record.behavioral_trust + magnitude)
        elif event_type == "negative_outcome":
            record.behavioral_trust = max(0.0, record.behavioral_trust - magnitude * 1.5)
        elif event_type == "betrayal":
            record.betrayal_history.append(datetime.now())
            record.affective_trust = max(0.0, record.affective_trust - 0.3)
            record.behavioral_trust = max(0.0, record.behavioral_trust - 0.4)
        elif event_type == "competence_display":
            record.cognitive_trust = min(1.0, record.cognitive_trust + magnitude)
        elif event_type == "emotional_support":
            record.affective_trust = min(1.0, record.affective_trust + magnitude)

        record.last_interaction = datetime.now()

    def propagate_reputation(self, agent_id: str, event: Dict[str, Any]):
        """Propagate reputation change through gossip network"""
        delta = event.get("reputation_delta", 0.0)
        source = event.get("source", "unknown")

        current = self.reputation.get(agent_id, 0.5)
        self.reputation[agent_id] = max(0.0, min(1.0, current + delta))

        # Spread through gossip network
        for gossiper in self.gossip_network.get(source, set()):
            # Gossip gets diluted
            gossip_delta = delta * 0.5
            gossiper_view = self.get_trust(gossiper, agent_id)
            gossiper_view.cognitive_trust += gossip_delta * 0.3

    def get_reputation(self, agent_id: str) -> float:
        """Get global reputation score"""
        return self.reputation.get(agent_id, 0.5)


# ============================================================================
# GROUP DYNAMICS
# ============================================================================

class GroupRole(Enum):
    """Roles within a group"""
    LEADER = "leader"
    CHALLENGER = "challenger"
    FOLLOWER = "follower"
    OUTSIDER = "outsider"
    MEDIATOR = "mediator"
    SCAPEGOAT = "scapegoat"
    NEWCOMER = "newcomer"


@dataclass
class SocialIdentity:
    """Agent's social identity - group memberships and identification"""
    agent_id: str
    groups: Dict[str, float] = field(default_factory=dict)  # Group -> identification strength
    roles: Dict[str, GroupRole] = field(default_factory=dict)  # Group -> role
    salience: Dict[str, float] = field(default_factory=dict)  # How salient each identity is

    def get_ingroup(self) -> List[str]:
        """Get groups agent strongly identifies with"""
        return [g for g, strength in self.groups.items() if strength > 0.6]

    def get_primary_identity(self) -> Optional[str]:
        """Get most salient group identity"""
        if not self.salience:
            return None
        return max(self.salience.items(), key=lambda x: x[1])[0]


class GroupDynamicsEngine:
    """
    Simulates group dynamics including:
    - Coalition formation
    - Status hierarchies
    - Ingroup/outgroup dynamics
    - Collective decision-making
    """

    def __init__(self):
        self.groups: Dict[str, Set[str]] = {}  # Group -> members
        self.identities: Dict[str, SocialIdentity] = {}  # Agent -> identity
        self.status_hierarchies: Dict[str, Dict[str, float]] = {}  # Group -> agent -> status
        self.group_norms: Dict[str, Dict[str, float]] = {}  # Group -> norm -> strength
        self.coalitions: List[Set[str]] = []

    def add_to_group(self, agent_id: str, group_id: str, initial_status: float = 0.5):
        """Add agent to a group"""
        if group_id not in self.groups:
            self.groups[group_id] = set()
            self.status_hierarchies[group_id] = {}
            self.group_norms[group_id] = {}

        self.groups[group_id].add(agent_id)
        self.status_hierarchies[group_id][agent_id] = initial_status

        if agent_id not in self.identities:
            self.identities[agent_id] = SocialIdentity(agent_id=agent_id)

        self.identities[agent_id].groups[group_id] = 0.5  # Initial identification
        self.identities[agent_id].roles[group_id] = GroupRole.NEWCOMER

    def update_status(self, group_id: str, agent_id: str, delta: float):
        """Update agent's status within a group"""
        if group_id in self.status_hierarchies:
            current = self.status_hierarchies[group_id].get(agent_id, 0.5)
            self.status_hierarchies[group_id][agent_id] = max(0.0, min(1.0, current + delta))

            # Update roles based on status
            self._update_roles(group_id)

    def _update_roles(self, group_id: str):
        """Update roles based on status hierarchy"""
        if group_id not in self.status_hierarchies:
            return

        statuses = self.status_hierarchies[group_id]
        if not statuses:
            return

        sorted_agents = sorted(statuses.items(), key=lambda x: -x[1])

        # Assign roles
        for i, (agent_id, status) in enumerate(sorted_agents):
            if agent_id in self.identities:
                if i == 0 and status > 0.7:
                    self.identities[agent_id].roles[group_id] = GroupRole.LEADER
                elif i == 1 and status > 0.6:
                    self.identities[agent_id].roles[group_id] = GroupRole.CHALLENGER
                elif status < 0.2:
                    self.identities[agent_id].roles[group_id] = GroupRole.OUTSIDER
                else:
                    self.identities[agent_id].roles[group_id] = GroupRole.FOLLOWER

    def calculate_ingroup_bias(self, agent_id: str, target_id: str) -> float:
        """Calculate ingroup bias between two agents"""
        if agent_id not in self.identities or target_id not in self.identities:
            return 0.0

        agent_groups = set(self.identities[agent_id].groups.keys())
        target_groups = set(self.identities[target_id].groups.keys())

        shared_groups = agent_groups & target_groups

        if not shared_groups:
            return -0.2  # Slight outgroup bias

        # Bias strength based on identification strength
        total_bias = 0.0
        for group in shared_groups:
            agent_ident = self.identities[agent_id].groups.get(group, 0)
            target_ident = self.identities[target_id].groups.get(group, 0)
            total_bias += (agent_ident + target_ident) / 2

        return total_bias / len(shared_groups)

    def detect_coalitions(self, interaction_matrix: Dict[Tuple[str, str], float]) -> List[Set[str]]:
        """Detect emergent coalitions from interaction patterns"""
        # Simple clustering based on interaction strength
        agents = set()
        for (a1, a2) in interaction_matrix.keys():
            agents.add(a1)
            agents.add(a2)

        coalitions = []
        used = set()

        for agent in agents:
            if agent in used:
                continue

            coalition = {agent}
            for (a1, a2), strength in interaction_matrix.items():
                if strength > 0.6:
                    if a1 in coalition or a2 in coalition:
                        coalition.add(a1)
                        coalition.add(a2)

            if len(coalition) >= 2:
                coalitions.append(coalition)
                used.update(coalition)

        self.coalitions = coalitions
        return coalitions


# ============================================================================
# INTEGRATED PSYCHOLOGY SYSTEM
# ============================================================================

class PsychologyEngine:
    """
    Integrated psychology engine combining all systems.
    """

    def __init__(self):
        self.personalities: Dict[str, PersonalityProfile] = {}
        self.biases: Dict[str, BiasProfile] = {}
        self.tom: Dict[str, TheoryOfMind] = {}
        self.emotions: Dict[str, EmotionalState] = {}
        self.contagion = EmotionalContagion()
        self.influence = SocialInfluenceModel()
        self.trust = TrustNetwork()
        self.groups = GroupDynamicsEngine()

    def register_agent(self, agent_id: str, personality: PersonalityProfile = None,
                      archetype: str = None):
        """Register a new agent with psychological systems"""
        if personality:
            self.personalities[agent_id] = personality
        else:
            self.personalities[agent_id] = generate_random_personality(archetype)

        self.biases[agent_id] = BiasProfile()
        self.tom[agent_id] = TheoryOfMind(agent_id, self.personalities[agent_id])
        self.emotions[agent_id] = EmotionalState()

        # Set contagion properties based on personality
        susc = 0.5 + (self.personalities[agent_id].agreeableness - 0.5) * 0.5
        expr = 0.5 + (self.personalities[agent_id].extraversion - 0.5) * 0.5
        self.contagion.set_agent_properties(agent_id, susc, expr)

    def process_interaction(self, agent1: str, agent2: str,
                           interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Process an interaction between two agents"""
        results = {
            "emotional_changes": {},
            "belief_updates": {},
            "trust_changes": {},
            "influence_attempts": []
        }

        # Theory of Mind updates
        if agent1 in self.tom:
            self.tom[agent1].update_from_observation(agent2, interaction)
        if agent2 in self.tom:
            self.tom[agent2].update_from_observation(agent1, interaction)

        # Emotional contagion
        if agent1 in self.emotions and agent2 in self.emotions:
            intensity = interaction.get("intensity", 0.5)
            deltas1 = self.contagion.calculate_contagion(
                agent2, self.emotions[agent2],
                agent1, self.emotions[agent1],
                intensity
            )
            deltas2 = self.contagion.calculate_contagion(
                agent1, self.emotions[agent1],
                agent2, self.emotions[agent2],
                intensity
            )

            # Apply deltas
            for emotion, delta in deltas1.items():
                self.emotions[agent1].emotions[emotion] += delta
            for emotion, delta in deltas2.items():
                self.emotions[agent2].emotions[emotion] += delta

            results["emotional_changes"][agent1] = deltas1
            results["emotional_changes"][agent2] = deltas2

        # Trust updates
        event_type = interaction.get("type", "neutral")
        if event_type in ["help", "share", "support"]:
            self.trust.update_trust(agent1, agent2, {"type": "positive_outcome", "magnitude": 0.1})
            self.trust.update_trust(agent2, agent1, {"type": "positive_outcome", "magnitude": 0.1})
        elif event_type in ["conflict", "betray", "deceive"]:
            self.trust.update_trust(agent1, agent2, {"type": "negative_outcome", "magnitude": 0.15})
            self.trust.update_trust(agent2, agent1, {"type": "negative_outcome", "magnitude": 0.15})

        results["trust_changes"] = {
            f"{agent1}->{agent2}": self.trust.get_trust(agent1, agent2).overall_trust,
            f"{agent2}->{agent1}": self.trust.get_trust(agent2, agent1).overall_trust
        }

        return results

    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get complete psychological state of an agent"""
        return {
            "personality": self.personalities.get(agent_id, PersonalityProfile()).to_dict(),
            "emotions": {
                e.value: v for e, v in self.emotions.get(agent_id, EmotionalState()).emotions.items()
            },
            "mood": self.emotions.get(agent_id, EmotionalState()).mood,
            "reputation": self.trust.get_reputation(agent_id),
            "group_identities": self.groups.identities.get(agent_id, SocialIdentity(agent_id)).groups
        }


# Convenience functions
def create_psychology_engine() -> PsychologyEngine:
    """Create a new psychology engine"""
    return PsychologyEngine()


def demo_psychology():
    """Demonstrate psychology engine"""
    engine = create_psychology_engine()

    # Create some agents
    engine.register_agent("alice", archetype="leader")
    engine.register_agent("bob", archetype="intellectual")
    engine.register_agent("carol", archetype="empath")

    # Add them to a group
    engine.groups.add_to_group("alice", "research_team", 0.8)
    engine.groups.add_to_group("bob", "research_team", 0.6)
    engine.groups.add_to_group("carol", "research_team", 0.5)

    # Simulate an interaction
    interaction = {
        "type": "discussion",
        "topic": "project_direction",
        "intensity": 0.7,
        "statement": {"topic": "project_direction", "position": 0.8, "certainty": 0.9}
    }

    results = engine.process_interaction("alice", "bob", interaction)

    print("=== Psychology Demo ===")
    print(f"\nAlice's state: {engine.get_agent_state('alice')}")
    print(f"\nBob's state: {engine.get_agent_state('bob')}")
    print(f"\nInteraction results: {results}")

    # Check Theory of Mind
    alice_tom = engine.tom["alice"]
    bob_model = alice_tom.get_or_create_model("bob")
    print(f"\nAlice's model of Bob's beliefs: {bob_model.beliefs}")

    return engine


if __name__ == "__main__":
    demo_psychology()
