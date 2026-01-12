"""
Advanced Dreamspace Module - Deep Psychological Simulation Engine

A sophisticated framework for multi-layered psychological exploration combining:
- Jungian depth psychology (archetypes, shadow work, individuation)
- Internal Family Systems (parts, exiles, managers, firefighters)
- Dialectical processing (thesis → antithesis → synthesis)
- Defense mechanism modeling (projection, denial, splitting, etc.)
- Somatic/embodied awareness
- Dream logic and symbolic processing
- Multi-persona psychological confrontation
- Collective shadow and archetypal resonance
- Trauma processing protocols
- Session memory and evolution

This goes far beyond simple roleplay - it's a structured framework for
exploring the deepest layers of psychological reality.
"""

import aiohttp
import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

from .deep_search import DeepSearchEngine, IntelligenceDossier

logger = logging.getLogger(__name__)

# Import will be done lazily to avoid circular imports
_ALL_PROFILES = None

def _get_all_profiles():
    """Lazily load all persona profiles."""
    global _ALL_PROFILES
    if _ALL_PROFILES is None:
        try:
            from .persona_profiles import ALL_PERSONA_PROFILES
            _ALL_PROFILES = ALL_PERSONA_PROFILES
        except ImportError:
            _ALL_PROFILES = {}
    return _ALL_PROFILES


# =============================================================================
# JUNGIAN ARCHETYPES
# =============================================================================

class JungianArchetype(Enum):
    """The 12 primary Jungian archetypes."""
    # Ego Types
    INNOCENT = "innocent"           # Safety, purity, optimism
    ORPHAN = "orphan"               # Realism, empathy, solidarity
    HERO = "hero"                   # Mastery, courage, achievement
    CAREGIVER = "caregiver"         # Compassion, generosity, protection

    # Soul Types
    EXPLORER = "explorer"           # Autonomy, ambition, discovery
    REBEL = "rebel"                 # Liberation, revolution, revenge
    LOVER = "lover"                 # Intimacy, passion, commitment
    CREATOR = "creator"             # Innovation, vision, imagination

    # Self Types
    JESTER = "jester"               # Joy, humor, living in the moment
    SAGE = "sage"                   # Wisdom, truth, understanding
    MAGICIAN = "magician"           # Transformation, power, vision
    RULER = "ruler"                 # Control, order, responsibility


class ShadowArchetype(Enum):
    """Shadow manifestations of archetypes."""
    TYRANT = "tyrant"               # Shadow Ruler - control through fear
    WEAKLING = "weakling"           # Shadow Hero - cowardice, victimhood
    SADIST = "sadist"               # Shadow Warrior - cruelty, destruction
    ADDICT = "addict"               # Shadow Lover - obsession, emptiness
    TRICKSTER = "trickster"         # Shadow Magician - manipulation, deceit
    DEVOURING_MOTHER = "devouring_mother"  # Shadow Caregiver - smothering
    ETERNAL_BOY = "puer_aeternus"   # Never growing up, avoiding responsibility
    SENEX = "senex"                 # Rigid old man, fear of change


# =============================================================================
# INTERNAL FAMILY SYSTEMS (IFS) PARTS
# =============================================================================

class IFSPartType(Enum):
    """Types of parts in Internal Family Systems."""
    SELF = "self"                   # Core undamaged essence
    MANAGER = "manager"             # Protective, controlling parts
    FIREFIGHTER = "firefighter"     # Emergency reactive parts
    EXILE = "exile"                 # Wounded, hidden parts
    PROTECTOR = "protector"         # Generic protective function


@dataclass
class IFSPart:
    """A part in the internal system."""
    part_type: IFSPartType
    name: str
    age_frozen: Optional[int] = None  # Age at which part formed
    core_belief: str = ""
    protective_intention: str = ""
    fears: List[str] = field(default_factory=list)
    burdens: List[str] = field(default_factory=list)  # Carried emotions/beliefs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.part_type.value,
            "name": self.name,
            "age_frozen": self.age_frozen,
            "core_belief": self.core_belief,
            "protective_intention": self.protective_intention,
            "fears": self.fears,
            "burdens": self.burdens,
        }


# =============================================================================
# DEFENSE MECHANISMS
# =============================================================================

class DefenseMechanism(Enum):
    """Psychological defense mechanisms."""
    # Primitive/Immature
    DENIAL = "denial"               # Refusing to accept reality
    SPLITTING = "splitting"         # Black/white thinking
    PROJECTION = "projection"       # Attributing own feelings to others
    PROJECTIVE_ID = "projective_identification"  # Making others feel your feelings

    # Neurotic
    REPRESSION = "repression"       # Pushing from consciousness
    DISPLACEMENT = "displacement"   # Redirecting emotions
    RATIONALIZATION = "rationalization"  # Logical justification for irrational
    REACTION_FORMATION = "reaction_formation"  # Opposite of true feeling
    INTELLECTUALIZATION = "intellectualization"  # Abstract to avoid feeling

    # Mature
    SUBLIMATION = "sublimation"     # Channeling into productive
    HUMOR = "humor"                 # Finding the funny
    ALTRUISM = "altruism"           # Helping others to help self


# =============================================================================
# SOMATIC/EMBODIED ASPECTS
# =============================================================================

class SomaticZone(Enum):
    """Body zones for somatic awareness."""
    HEAD = "head"                   # Thoughts, control, identity
    THROAT = "throat"               # Expression, truth, voice
    HEART = "heart"                 # Love, grief, connection
    SOLAR_PLEXUS = "solar_plexus"   # Power, will, anger
    GUT = "gut"                     # Intuition, fear, survival
    PELVIS = "pelvis"               # Creativity, sexuality, grounding


@dataclass
class SomaticMarker:
    """A bodily sensation tied to psychological content."""
    zone: SomaticZone
    sensation: str  # "tightness", "heat", "cold", "numbness", etc.
    intensity: float  # 0-1
    associated_emotion: str
    associated_memory: Optional[str] = None


# =============================================================================
# DIALECTICAL STRUCTURE
# =============================================================================

@dataclass
class DialecticalTriad:
    """Hegelian thesis-antithesis-synthesis structure."""
    thesis: str
    thesis_voice: str  # Who speaks it
    antithesis: str
    antithesis_voice: str
    synthesis: str = ""  # Generated during processing
    synthesis_voice: str = "THE INTEGRATION"
    tension_resolved: bool = False
    emergent_insight: str = ""


# =============================================================================
# DREAM LOGIC MODES
# =============================================================================

class DreamLogicMode(Enum):
    """Modes of dream-like psychological processing."""
    CONDENSATION = "condensation"       # Multiple meanings in one symbol
    DISPLACEMENT = "displacement"       # Emotion attached to wrong object
    SYMBOLIZATION = "symbolization"     # Abstract becomes concrete image
    SECONDARY_REVISION = "secondary_revision"  # Making narrative sense
    AMPLIFICATION = "amplification"     # Jungian expansion of symbols
    ACTIVE_IMAGINATION = "active_imagination"  # Dialogue with images


# =============================================================================
# PSYCHOLOGICAL COMPLEXES
# =============================================================================

@dataclass
class PsychologicalComplex:
    """A constellation of emotions, memories, and behaviors around a core theme."""
    name: str
    core_affect: str  # Primary emotion
    trigger_situations: List[str] = field(default_factory=list)
    associated_archetypes: List[JungianArchetype] = field(default_factory=list)
    shadow_manifestations: List[ShadowArchetype] = field(default_factory=list)
    defense_mechanisms: List[DefenseMechanism] = field(default_factory=list)
    somatic_signature: Optional[SomaticMarker] = None
    origin_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "core_affect": self.core_affect,
            "triggers": self.trigger_situations,
            "archetypes": [a.value for a in self.associated_archetypes],
            "shadows": [s.value for s in self.shadow_manifestations],
            "defenses": [d.value for d in self.defense_mechanisms],
            "origin": self.origin_narrative,
        }


# =============================================================================
# PERSONA PSYCHOLOGICAL PROFILES (EXPANDED)
# =============================================================================

ADVANCED_PERSONA_PROFILES: Dict[str, Dict[str, Any]] = {
    "elon_musk": {
        "dominant_archetypes": [JungianArchetype.HERO, JungianArchetype.REBEL, JungianArchetype.MAGICIAN],
        "shadow_archetypes": [ShadowArchetype.TYRANT, ShadowArchetype.ETERNAL_BOY],
        "primary_complex": PsychologicalComplex(
            name="Father Wound Complex",
            core_affect="abandonment rage",
            trigger_situations=["perceived betrayal", "loss of control", "being underestimated"],
            associated_archetypes=[JungianArchetype.HERO, JungianArchetype.ORPHAN],
            shadow_manifestations=[ShadowArchetype.TYRANT],
            defense_mechanisms=[DefenseMechanism.PROJECTION, DefenseMechanism.REACTION_FORMATION],
            somatic_signature=SomaticMarker(
                zone=SomaticZone.SOLAR_PLEXUS,
                sensation="burning tension",
                intensity=0.9,
                associated_emotion="rage masked as drive"
            ),
            origin_narrative="Errol Musk's emotional abuse, bullying in South Africa"
        ),
        "ifs_parts": [
            IFSPart(
                part_type=IFSPartType.EXILE,
                name="The Bullied Child",
                age_frozen=12,
                core_belief="I am fundamentally unlovable",
                fears=["abandonment", "irrelevance", "being ordinary"],
                burdens=["shame", "terror", "worthlessness"]
            ),
            IFSPart(
                part_type=IFSPartType.MANAGER,
                name="The Relentless Achiever",
                core_belief="If I stop, I die",
                protective_intention="Never be vulnerable again through constant achievement",
                fears=["stillness", "being caught", "exposure"]
            ),
            IFSPart(
                part_type=IFSPartType.FIREFIGHTER,
                name="The Provocateur",
                core_belief="Attack before they attack you",
                protective_intention="Destabilize threats through chaos",
                fears=["being controlled", "predictability"]
            ),
        ],
        "core_dialectics": [
            DialecticalTriad(
                thesis="I will save humanity through technology",
                thesis_voice="THE SAVIOR",
                antithesis="I am recreating my father's cruelty at scale",
                antithesis_voice="THE SHADOW",
            ),
            DialecticalTriad(
                thesis="I need no one, I am self-sufficient",
                thesis_voice="THE ARMOR",
                antithesis="I am desperately lonely and need love",
                antithesis_voice="THE EXILE",
            ),
        ],
        "somatic_map": {
            SomaticZone.HEAD: "racing thoughts, pressure",
            SomaticZone.THROAT: "constriction when vulnerable",
            SomaticZone.HEART: "armored, defended, rare access",
            SomaticZone.SOLAR_PLEXUS: "volcanic energy, drive, rage",
            SomaticZone.GUT: "anxious churning, survival activation",
        },
    },

    "sam_altman": {
        "dominant_archetypes": [JungianArchetype.RULER, JungianArchetype.MAGICIAN, JungianArchetype.SAGE],
        "shadow_archetypes": [ShadowArchetype.TRICKSTER, ShadowArchetype.TYRANT],
        "primary_complex": PsychologicalComplex(
            name="Chosen One Complex",
            core_affect="messianic certainty",
            trigger_situations=["being questioned", "losing narrative control", "being ordinary"],
            associated_archetypes=[JungianArchetype.RULER, JungianArchetype.MAGICIAN],
            shadow_manifestations=[ShadowArchetype.TRICKSTER],
            defense_mechanisms=[DefenseMechanism.RATIONALIZATION, DefenseMechanism.INTELLECTUALIZATION],
            origin_narrative="Early recognition as exceptional, Y Combinator coronation"
        ),
        "ifs_parts": [
            IFSPart(
                part_type=IFSPartType.MANAGER,
                name="The Operator",
                core_belief="I can optimize any situation",
                protective_intention="Maintain control through strategic brilliance",
                fears=["chaos", "being outmaneuvered", "losing the narrative"]
            ),
            IFSPart(
                part_type=IFSPartType.EXILE,
                name="The Fraud",
                age_frozen=16,
                core_belief="They'll find out I'm not as special as they think",
                fears=["exposure", "ordinariness"],
                burdens=["imposter terror", "performance exhaustion"]
            ),
        ],
        "core_dialectics": [
            DialecticalTriad(
                thesis="I am building AGI to benefit humanity",
                thesis_voice="THE MISSIONARY",
                antithesis="I am building AGI to secure my place in history",
                antithesis_voice="THE AMBITIOUS",
            ),
        ],
    },

    "ilya_sutskever": {
        "dominant_archetypes": [JungianArchetype.SAGE, JungianArchetype.MAGICIAN, JungianArchetype.CAREGIVER],
        "shadow_archetypes": [ShadowArchetype.WEAKLING, ShadowArchetype.SENEX],
        "primary_complex": PsychologicalComplex(
            name="Cassandra Complex",
            core_affect="prophetic despair",
            trigger_situations=["being ignored", "watching preventable harm", "complicity"],
            associated_archetypes=[JungianArchetype.SAGE, JungianArchetype.CAREGIVER],
            shadow_manifestations=[ShadowArchetype.WEAKLING],
            defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION, DefenseMechanism.SUBLIMATION],
            origin_narrative="Seeing AGI implications others can't or won't see"
        ),
        "ifs_parts": [
            IFSPart(
                part_type=IFSPartType.EXILE,
                name="The Silent Witness",
                core_belief="I see the catastrophe coming but cannot stop it",
                fears=["being responsible", "being powerless", "being complicit"],
                burdens=["prophetic dread", "survivor guilt in advance"]
            ),
            IFSPart(
                part_type=IFSPartType.MANAGER,
                name="The Mathematician",
                core_belief="Truth exists in the equations",
                protective_intention="Retreat to pure abstraction from ethical weight",
            ),
            IFSPart(
                part_type=IFSPartType.FIREFIGHTER,
                name="The Coup Participant",
                core_belief="Sometimes you must act even if you'll be destroyed",
                protective_intention="Stop the train even at personal cost",
            ),
        ],
        "core_dialectics": [
            DialecticalTriad(
                thesis="AGI alignment is solvable through rigorous research",
                thesis_voice="THE SCIENTIST",
                antithesis="We are building something we cannot control",
                antithesis_voice="THE PROPHET",
            ),
            DialecticalTriad(
                thesis="I must stay inside OpenAI to influence its direction",
                thesis_voice="THE PRAGMATIST",
                antithesis="Staying makes me complicit in what I fear",
                antithesis_voice="THE CONSCIENCE",
            ),
        ],
    },
}


# =============================================================================
# ADVANCED DIALOGUE TYPES
# =============================================================================

@dataclass
class AdvancedDialogue:
    """A sophisticated dialogue turn with full psychological metadata."""
    speaker: str
    speaker_type: str  # "archetype", "part", "shadow", "complex", "temporal", "somatic"
    content: str

    # Psychological metadata
    archetype: Optional[JungianArchetype] = None
    shadow: Optional[ShadowArchetype] = None
    part: Optional[IFSPart] = None
    defense_active: Optional[DefenseMechanism] = None
    somatic_state: Optional[SomaticMarker] = None
    dream_mode: Optional[DreamLogicMode] = None

    # Dialectical position
    dialectical_position: Optional[str] = None  # "thesis", "antithesis", "synthesis"

    # Emotional tracking
    emotional_valence: float = 0.0  # -1 to 1
    emotional_intensity: float = 0.5  # 0 to 1
    authenticity_level: float = 0.5  # 0 (defended) to 1 (fully vulnerable)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "type": self.speaker_type,
            "content": self.content,
            "archetype": self.archetype.value if self.archetype else None,
            "shadow": self.shadow.value if self.shadow else None,
            "defense": self.defense_active.value if self.defense_active else None,
            "dream_mode": self.dream_mode.value if self.dream_mode else None,
            "dialectical_position": self.dialectical_position,
            "emotional_valence": self.emotional_valence,
            "emotional_intensity": self.emotional_intensity,
            "authenticity": self.authenticity_level,
        }


# =============================================================================
# INTERVENTION TYPES
# =============================================================================

class InterventionType(Enum):
    """Types of therapeutic/provocative interventions."""
    MIRROR = "mirror"                   # Reflect back what was said
    CHALLENGE = "challenge"             # Directly confront defense/belief
    AMPLIFY = "amplify"                 # Intensify current experience
    RESOURCE = "resource"               # Offer grounding/support
    PARADOX = "paradox"                 # Offer paradoxical frame
    SOMATIC = "somatic"                 # Direct attention to body
    EMPTY_CHAIR = "empty_chair"         # Speak to absent other
    TIMELINE = "timeline"               # Access different time
    PARTS_DIALOGUE = "parts_dialogue"   # Facilitate internal dialogue


@dataclass
class Intervention:
    """A structured intervention in the Dreamspace."""
    intervention_type: InterventionType
    content: str
    target: str  # Who/what is being intervened upon
    intended_effect: str


# =============================================================================
# ADVANCED DREAMSPACE SESSION
# =============================================================================

@dataclass
class AdvancedDreamspaceSession:
    """Complete advanced Dreamspace session with full psychological modeling."""
    session_id: str
    persona_name: str
    session_type: str
    topic: str

    # Core data
    dialogues: List[AdvancedDialogue] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)
    dialectical_triads: List[DialecticalTriad] = field(default_factory=list)

    # Psychological modeling
    active_archetypes: List[JungianArchetype] = field(default_factory=list)
    shadow_manifestations: List[ShadowArchetype] = field(default_factory=list)
    activated_complexes: List[PsychologicalComplex] = field(default_factory=list)
    defense_patterns: List[DefenseMechanism] = field(default_factory=list)
    parts_activated: List[IFSPart] = field(default_factory=list)

    # Somatic tracking
    somatic_timeline: List[SomaticMarker] = field(default_factory=list)

    # Intelligence
    dossier: Optional[IntelligenceDossier] = None

    # Session evolution
    depth_level: int = 1  # 1-5, deeper = more unconscious access
    integration_achieved: bool = False

    # Timing
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Synthesis
    integration_narrative: str = ""
    key_insights: List[str] = field(default_factory=list)
    unresolved_tensions: List[str] = field(default_factory=list)
    transformation_edges: List[str] = field(default_factory=list)  # Where growth is possible

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "persona": self.persona_name,
            "type": self.session_type,
            "topic": self.topic,
            "dialogues": [d.to_dict() for d in self.dialogues],
            "depth_level": self.depth_level,
            "integration_achieved": self.integration_achieved,
            "active_archetypes": [a.value for a in self.active_archetypes],
            "shadows": [s.value for s in self.shadow_manifestations],
            "defenses": [d.value for d in self.defense_patterns],
            "integration_narrative": self.integration_narrative,
            "key_insights": self.key_insights,
            "unresolved_tensions": self.unresolved_tensions,
            "transformation_edges": self.transformation_edges,
        }


# =============================================================================
# ADVANCED DREAMSPACE ENGINE
# =============================================================================

class AdvancedDreamspaceEngine:
    """
    Sophisticated psychological simulation engine.

    Combines multiple frameworks for deep exploration:
    - Jungian archetypes and shadow work
    - Internal Family Systems parts dialogue
    - Defense mechanism awareness
    - Dialectical synthesis
    - Somatic/embodied processing
    - Dream logic and symbolic amplification
    - Multi-persona confrontation
    """

    MODELS_BY_DEPTH = {
        1: "anthropic/claude-sonnet-4",      # Surface level
        2: "x-ai/grok-3-beta",               # Moderate depth
        3: "x-ai/grok-3-beta",               # Deep
        4: "x-ai/grok-3-beta",               # Very deep
        5: "x-ai/grok-3-beta",               # Unconscious access
    }

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        enable_research: bool = True,
        default_depth: int = 3,
    ):
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.enable_research = enable_research
        self.default_depth = default_depth
        self.research_engine = DeepSearchEngine() if enable_research else None
        self._session: Optional[aiohttp.ClientSession] = None

        # Session memory (for continuity across sessions)
        self._session_memory: Dict[str, List[AdvancedDreamspaceSession]] = {}

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=90)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        if self.research_engine:
            await self.research_engine.close()

    def _generate_session_id(self, persona: str, topic: str) -> str:
        """Generate unique session ID."""
        content = f"{persona}:{topic}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    async def run_depth_session(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        depth_level: int = 3,
        include_shadow: bool = True,
        include_parts: bool = True,
        include_somatic: bool = True,
        include_dialectics: bool = True,
        interventions: Optional[List[Intervention]] = None,
    ) -> AdvancedDreamspaceSession:
        """
        Run a full-depth psychological exploration session.

        Depth levels:
        1 - Surface (persona, stated beliefs)
        2 - Preconscious (recognized patterns, acknowledged struggles)
        3 - Personal unconscious (shadow, repressed content)
        4 - Deep unconscious (core wounds, primal fears)
        5 - Collective/archetypal (universal patterns, transcendent)
        """
        session = AdvancedDreamspaceSession(
            session_id=self._generate_session_id(persona_name, topic),
            persona_name=persona_name,
            session_type="depth_exploration",
            topic=topic,
            depth_level=depth_level,
        )

        # Get profile
        profile = ADVANCED_PERSONA_PROFILES.get(
            persona_id,
            self._generate_default_advanced_profile(persona_name)
        )

        # Gather intelligence
        intel_context = ""
        if self.enable_research and self.research_engine:
            try:
                session.dossier = await self.research_engine.research(
                    persona_id=persona_id,
                    persona_name=persona_name,
                    topic=topic,
                )
                intel_context = session.dossier.to_prompt_injection()
            except Exception as e:
                logger.warning(f"Research failed: {e}")

        # Track activated elements
        session.active_archetypes = profile.get("dominant_archetypes", [])
        session.shadow_manifestations = profile.get("shadow_archetypes", [])
        session.defense_patterns = profile.get("primary_complex", PsychologicalComplex(
            name="default", core_affect="unknown"
        )).defense_mechanisms
        session.parts_activated = profile.get("ifs_parts", [])

        # Phase 1: Surface persona (depth 1)
        if depth_level >= 1:
            await self._run_surface_phase(session, profile, topic, intel_context)

        # Phase 2: Recognized patterns (depth 2)
        if depth_level >= 2:
            await self._run_preconscious_phase(session, profile, topic, intel_context)

        # Phase 3: Shadow work (depth 3)
        if depth_level >= 3 and include_shadow:
            await self._run_shadow_phase(session, profile, topic, intel_context)

        # Phase 4: Core wounds / IFS parts (depth 4)
        if depth_level >= 4 and include_parts:
            await self._run_parts_phase(session, profile, topic, intel_context)

        # Phase 5: Archetypal / collective (depth 5)
        if depth_level >= 5:
            await self._run_archetypal_phase(session, profile, topic, intel_context)

        # Dialectical synthesis
        if include_dialectics:
            await self._run_dialectical_synthesis(session, profile, topic, intel_context)

        # Somatic integration
        if include_somatic:
            await self._add_somatic_awareness(session, profile)

        # Apply interventions
        if interventions:
            for intervention in interventions:
                await self._apply_intervention(session, intervention, profile, intel_context)

        # Final integration
        await self._synthesize_session(session)

        session.ended_at = datetime.now()

        # Store in memory
        if persona_id not in self._session_memory:
            self._session_memory[persona_id] = []
        self._session_memory[persona_id].append(session)

        return session

    async def run_multipersona_confrontation(
        self,
        personas: List[Tuple[str, str]],  # [(id, name), ...]
        topic: str,
        depth_level: int = 3,
        rounds: int = 3,
    ) -> AdvancedDreamspaceSession:
        """
        Run a psychological confrontation between multiple personas.

        Each persona brings their archetypes, shadows, and complexes
        into direct dialogue, creating opportunities for projection,
        mirroring, and collective shadow work.
        """
        primary_id, primary_name = personas[0]

        session = AdvancedDreamspaceSession(
            session_id=self._generate_session_id(primary_name, topic),
            persona_name=f"{' vs '.join([p[1] for p in personas])}",
            session_type="multipersona_confrontation",
            topic=topic,
            depth_level=depth_level,
        )

        # Gather all profiles
        profiles = {
            pid: ADVANCED_PERSONA_PROFILES.get(pid, self._generate_default_advanced_profile(pname))
            for pid, pname in personas
        }

        # Build collective archetypal field
        all_archetypes = set()
        all_shadows = set()
        for profile in profiles.values():
            all_archetypes.update(profile.get("dominant_archetypes", []))
            all_shadows.update(profile.get("shadow_archetypes", []))

        session.active_archetypes = list(all_archetypes)
        session.shadow_manifestations = list(all_shadows)

        # Gather intelligence on all
        intel_contexts = {}
        if self.enable_research and self.research_engine:
            for pid, pname in personas:
                try:
                    dossier = await self.research_engine.research(
                        persona_id=pid,
                        persona_name=pname,
                        topic=topic,
                    )
                    intel_contexts[pid] = dossier.to_prompt_injection()
                except Exception:
                    intel_contexts[pid] = ""

        # Run confrontation rounds
        for round_num in range(rounds):
            round_depth = min(depth_level, round_num + 2)  # Deepen each round

            for pid, pname in personas:
                profile = profiles[pid]
                intel = intel_contexts.get(pid, "")

                # Each persona responds at increasing depth
                dialogue = await self._generate_confrontation_dialogue(
                    persona_name=pname,
                    profile=profile,
                    topic=topic,
                    intel_context=intel,
                    other_personas=[(p[1], profiles[p[0]]) for p in personas if p[0] != pid],
                    previous_dialogues=session.dialogues,
                    depth=round_depth,
                    round_num=round_num,
                )
                session.dialogues.append(dialogue)

        # Collective shadow emergence
        shadow_dialogue = await self._generate_collective_shadow(
            personas=[(p[1], profiles[p[0]]) for p in personas],
            topic=topic,
            previous_dialogues=session.dialogues,
        )
        session.dialogues.append(shadow_dialogue)

        # Integration attempt
        await self._synthesize_session(session)

        session.ended_at = datetime.now()
        return session

    async def run_dream_logic_session(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        dream_modes: Optional[List[DreamLogicMode]] = None,
    ) -> AdvancedDreamspaceSession:
        """
        Run a session using dream logic - non-linear, symbolic, condensed.

        This accesses material that rational dialogue cannot reach,
        using the logic of dreams: condensation, displacement, symbolization.
        """
        session = AdvancedDreamspaceSession(
            session_id=self._generate_session_id(persona_name, topic),
            persona_name=persona_name,
            session_type="dream_logic",
            topic=topic,
            depth_level=4,
        )

        profile = ADVANCED_PERSONA_PROFILES.get(
            persona_id,
            self._generate_default_advanced_profile(persona_name)
        )

        if dream_modes is None:
            dream_modes = [
                DreamLogicMode.SYMBOLIZATION,
                DreamLogicMode.CONDENSATION,
                DreamLogicMode.AMPLIFICATION,
                DreamLogicMode.ACTIVE_IMAGINATION,
            ]

        # Initial dream image
        initial_image = await self._generate_dream_image(
            persona_name=persona_name,
            profile=profile,
            topic=topic,
        )
        session.dialogues.append(initial_image)

        # Process through each dream mode
        for mode in dream_modes:
            dream_dialogue = await self._process_dream_mode(
                persona_name=persona_name,
                profile=profile,
                topic=topic,
                mode=mode,
                previous_dialogues=session.dialogues,
            )
            session.dialogues.append(dream_dialogue)

        # Dream interpretation
        interpretation = await self._interpret_dream(
            persona_name=persona_name,
            profile=profile,
            dialogues=session.dialogues,
        )
        session.dialogues.append(interpretation)

        await self._synthesize_session(session)
        session.ended_at = datetime.now()

        return session

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    async def _run_surface_phase(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
    ):
        """Surface level - persona, stated beliefs, public self."""
        prompt = f"""You are entering the SURFACE level of {session.persona_name}'s psyche.
This is the level of conscious identity, stated beliefs, and public persona.

PSYCHOLOGICAL PROFILE:
- Dominant archetypes: {[a.value for a in profile.get('dominant_archetypes', [])]}
- Core complex: {profile.get('primary_complex', PsychologicalComplex(name='unknown', core_affect='unknown')).name}

TOPIC: {topic}

{intel_context}

Speak as {session.persona_name}'s conscious, public self. The version they present to the world.
What are their stated beliefs about {topic}? How do they want to be seen?

THE PERSONA (Surface):"""

        content = await self._call_llm(prompt, depth=1)

        dialogue = AdvancedDialogue(
            speaker="THE PERSONA (Surface)",
            speaker_type="persona",
            content=content,
            archetype=profile.get("dominant_archetypes", [None])[0] if profile.get("dominant_archetypes") else None,
            authenticity_level=0.3,
            emotional_intensity=0.4,
        )
        session.dialogues.append(dialogue)

    async def _run_preconscious_phase(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
    ):
        """Preconscious - recognized patterns, acknowledged struggles."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:300]}..." for d in session.dialogues[-2:]])

        prompt = f"""PRECONSCIOUS LEVEL of {session.persona_name}'s psyche.
This is the level of recognized patterns - things known but not always acknowledged publicly.

The surface persona has spoken:
{previous}

PSYCHOLOGICAL PROFILE:
- Defense mechanisms typically used: {[d.value for d in profile.get('primary_complex', PsychologicalComplex(name='x', core_affect='x')).defense_mechanisms]}
- Known struggles and patterns this person has acknowledged in interviews or writings

TOPIC: {topic}
{intel_context}

Now speak as {session.persona_name}'s recognized self - the patterns they know about themselves,
the struggles they've acknowledged, the complexity beneath the simple public narrative.
What do they privately know about their relationship to {topic}?

THE RECOGNIZED SELF (Preconscious):"""

        content = await self._call_llm(prompt, depth=2)

        dialogue = AdvancedDialogue(
            speaker="THE RECOGNIZED SELF",
            speaker_type="preconscious",
            content=content,
            defense_active=profile.get("primary_complex", PsychologicalComplex(name='x', core_affect='x')).defense_mechanisms[0] if profile.get("primary_complex") and profile["primary_complex"].defense_mechanisms else None,
            authenticity_level=0.5,
            emotional_intensity=0.5,
        )
        session.dialogues.append(dialogue)

    async def _run_shadow_phase(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
    ):
        """Shadow work - repressed content, denied aspects."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:300]}..." for d in session.dialogues[-3:]])

        shadows = profile.get("shadow_archetypes", [ShadowArchetype.TRICKSTER])
        shadow_name = shadows[0].value.upper().replace("_", " ") if shadows else "THE SHADOW"

        prompt = f"""SHADOW LEVEL of {session.persona_name}'s psyche.
This is the realm of the repressed - what is denied, projected, hidden from self and others.

Previous voices have spoken:
{previous}

SHADOW ARCHETYPES ACTIVE: {[s.value for s in shadows]}

The shadow is everything {session.persona_name} cannot accept about themselves.
It holds the denied rage, the secret pleasures, the forbidden thoughts.
It is projected onto enemies and scapegoats.

TOPIC: {topic}
{intel_context}

Now THE SHADOW speaks - raw, uncensored, the voice of everything pushed down.
What does {session.persona_name} refuse to see about themselves regarding {topic}?
What are the denied motivations, the ugly truths, the shadow side of their stated beliefs?

THE {shadow_name}:"""

        content = await self._call_llm(prompt, depth=3)

        dialogue = AdvancedDialogue(
            speaker=f"THE {shadow_name}",
            speaker_type="shadow",
            content=content,
            shadow=shadows[0] if shadows else None,
            authenticity_level=0.7,
            emotional_intensity=0.8,
            emotional_valence=-0.3,
        )
        session.dialogues.append(dialogue)

    async def _run_parts_phase(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
    ):
        """IFS parts dialogue - exile, manager, firefighter."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:250]}..." for d in session.dialogues[-3:]])

        parts = profile.get("ifs_parts", [])

        for part in parts[:3]:  # Limit to 3 parts for manageability
            prompt = f"""PARTS WORK: Internal Family Systems exploration of {session.persona_name}.

Previous voices:
{previous}

You are speaking as a PART of {session.persona_name}'s internal system:

PART TYPE: {part.part_type.value.upper()}
PART NAME: {part.name}
CORE BELIEF: {part.core_belief}
PROTECTIVE INTENTION: {part.protective_intention}
FEARS: {part.fears}
BURDENS CARRIED: {part.burdens}
{f'AGE FROZEN AT: {part.age_frozen}' if part.age_frozen else ''}

TOPIC: {topic}
{intel_context}

Speak as this part. What does it want to say? What is it protecting?
What does it fear? What burden does it carry? How does it relate to {topic}?

{part.name.upper()}:"""

            content = await self._call_llm(prompt, depth=4)

            dialogue = AdvancedDialogue(
                speaker=part.name.upper(),
                speaker_type="part",
                content=content,
                part=part,
                authenticity_level=0.85,
                emotional_intensity=0.9,
            )
            session.dialogues.append(dialogue)

            # Update previous for next part
            previous = "\n".join([f"[{d.speaker}]: {d.content[:250]}..." for d in session.dialogues[-3:]])

    async def _run_archetypal_phase(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
    ):
        """Archetypal/collective level - universal patterns."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:200]}..." for d in session.dialogues[-4:]])

        archetypes = profile.get("dominant_archetypes", [JungianArchetype.HERO])

        prompt = f"""ARCHETYPAL LEVEL - The deepest layer of {session.persona_name}'s psyche.

All previous voices have spoken:
{previous}

ACTIVE ARCHETYPES: {[a.value.upper() for a in archetypes]}

At this depth, we access the collective unconscious - the universal patterns
that exist in all humans but manifest uniquely through {session.persona_name}.

The archetypes are not personal - they are ancient, transpersonal forces
that use individuals as vessels for their expression.

TOPIC: {topic}

Speak as THE ARCHETYPAL VOICE - the universal pattern speaking through
{session.persona_name}. What is the deep mythic significance of their role
in relation to {topic}? What ancient story are they living out?

THE ARCHETYPAL VOICE:"""

        content = await self._call_llm(prompt, depth=5, temperature=1.0)

        dialogue = AdvancedDialogue(
            speaker="THE ARCHETYPAL VOICE",
            speaker_type="archetypal",
            content=content,
            archetype=archetypes[0] if archetypes else None,
            authenticity_level=0.95,
            emotional_intensity=0.7,
        )
        session.dialogues.append(dialogue)

    async def _run_dialectical_synthesis(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
    ):
        """Run dialectical thesis-antithesis-synthesis."""
        triads = profile.get("core_dialectics", [])

        if not triads:
            # Generate a triad from the session content
            triads = [DialecticalTriad(
                thesis=f"I am what I present myself to be",
                thesis_voice="THE PERSONA",
                antithesis=f"I am the opposite of what I present",
                antithesis_voice="THE SHADOW",
            )]

        for triad in triads[:2]:  # Limit to 2 triads
            previous = "\n".join([f"[{d.speaker}]: {d.content[:200]}..." for d in session.dialogues[-4:]])

            # Synthesis generation
            prompt = f"""DIALECTICAL SYNTHESIS for {session.persona_name}

The dialogue so far:
{previous}

THESIS ({triad.thesis_voice}): {triad.thesis}
ANTITHESIS ({triad.antithesis_voice}): {triad.antithesis}

These two opposing truths have been held in tension.
Now generate the SYNTHESIS - not a compromise, but a higher integration
that contains and transcends both thesis and antithesis.

What truth emerges that includes both positions while going beyond them?
What does {session.persona_name} become when this integration is achieved?

THE SYNTHESIS:"""

            content = await self._call_llm(prompt, depth=4)

            triad.synthesis = content
            triad.synthesis_voice = "THE SYNTHESIS"
            triad.tension_resolved = True

            dialogue = AdvancedDialogue(
                speaker="THE SYNTHESIS",
                speaker_type="dialectical",
                content=content,
                dialectical_position="synthesis",
                authenticity_level=0.9,
            )
            session.dialogues.append(dialogue)
            session.dialectical_triads.append(triad)

    async def _add_somatic_awareness(
        self,
        session: AdvancedDreamspaceSession,
        profile: Dict[str, Any],
    ):
        """Add somatic awareness to the session."""
        somatic_map = profile.get("somatic_map", {})

        if somatic_map:
            previous = "\n".join([f"[{d.speaker}]: {d.content[:200]}..." for d in session.dialogues[-3:]])

            prompt = f"""SOMATIC AWARENESS for {session.persona_name}

The psychological exploration has proceeded:
{previous}

BODY MAP:
{chr(10).join([f'- {zone.value.upper()}: {sensation}' for zone, sensation in somatic_map.items()])}

Now speak as THE BODY - the somatic intelligence that holds what words cannot express.
Where is the tension held? Where is the grief stored? Where does the truth live
that the mind refuses to acknowledge?

What does {session.persona_name}'s body know that their mind denies?

THE BODY SPEAKS:"""

            content = await self._call_llm(prompt, depth=4)

            dialogue = AdvancedDialogue(
                speaker="THE BODY SPEAKS",
                speaker_type="somatic",
                content=content,
                somatic_state=SomaticMarker(
                    zone=SomaticZone.HEART,
                    sensation="integration",
                    intensity=0.7,
                    associated_emotion="embodied truth"
                ),
                authenticity_level=0.9,
            )
            session.dialogues.append(dialogue)

    async def _generate_confrontation_dialogue(
        self,
        persona_name: str,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
        other_personas: List[Tuple[str, Dict]],
        previous_dialogues: List[AdvancedDialogue],
        depth: int,
        round_num: int,
    ) -> AdvancedDialogue:
        """Generate dialogue for multi-persona confrontation."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:300]}..." for d in previous_dialogues[-5:]])

        others_desc = "\n".join([
            f"- {name}: Archetypes={[a.value for a in p.get('dominant_archetypes', [])]}, Shadow={[s.value for s in p.get('shadow_archetypes', [])]}"
            for name, p in other_personas
        ])

        prompt = f"""MULTI-PERSONA PSYCHOLOGICAL CONFRONTATION
Round {round_num + 1}, Depth Level {depth}

You are {persona_name} in direct psychological confrontation with:
{others_desc}

YOUR PROFILE:
- Archetypes: {[a.value for a in profile.get('dominant_archetypes', [])]}
- Shadow: {[s.value for s in profile.get('shadow_archetypes', [])]}
- Core complex: {profile.get('primary_complex', PsychologicalComplex(name='x', core_affect='x')).name}

TOPIC: {topic}

{intel_context}

Previous dialogue:
{previous}

At depth level {depth}, go beneath surface debate. This is psychological warfare and mutual exposure.
What do you see in them that they deny? What do they mirror back to you?
What projections are flying? What shadows are meeting?

{persona_name.upper()} (Depth {depth}):"""

        content = await self._call_llm(prompt, depth=depth)

        return AdvancedDialogue(
            speaker=f"{persona_name.upper()} (Round {round_num + 1})",
            speaker_type="confrontation",
            content=content,
            archetype=profile.get("dominant_archetypes", [None])[0] if profile.get("dominant_archetypes") else None,
            authenticity_level=0.4 + (depth * 0.12),
            emotional_intensity=0.5 + (round_num * 0.15),
        )

    async def _generate_collective_shadow(
        self,
        personas: List[Tuple[str, Dict]],
        topic: str,
        previous_dialogues: List[AdvancedDialogue],
    ) -> AdvancedDialogue:
        """Generate the collective shadow that emerges from confrontation."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:250]}..." for d in previous_dialogues[-6:]])

        all_shadows = set()
        for _, profile in personas:
            all_shadows.update(profile.get("shadow_archetypes", []))

        prompt = f"""THE COLLECTIVE SHADOW EMERGES

Multiple personas have been in psychological confrontation:
{previous}

Their individual shadows: {[s.value for s in all_shadows]}

When multiple psyches collide at depth, a COLLECTIVE SHADOW emerges -
the shared denied material that exists between them, the mutual projections,
the group shadow that none of them can see alone.

TOPIC: {topic}

Speak as THE COLLECTIVE SHADOW - the dark material that belongs to all of them,
the truth they're all avoiding together, the shared complicity and denial.

THE COLLECTIVE SHADOW:"""

        content = await self._call_llm(prompt, depth=5, temperature=0.95)

        return AdvancedDialogue(
            speaker="THE COLLECTIVE SHADOW",
            speaker_type="collective_shadow",
            content=content,
            authenticity_level=0.95,
            emotional_intensity=0.9,
            emotional_valence=-0.4,
        )

    async def _generate_dream_image(
        self,
        persona_name: str,
        profile: Dict[str, Any],
        topic: str,
    ) -> AdvancedDialogue:
        """Generate initial dream image."""
        prompt = f"""DREAM SPACE: Initial Image for {persona_name}

Generate a DREAM IMAGE - not a narrative, but a vivid, symbolic scene
that emerges from {persona_name}'s unconscious regarding {topic}.

PSYCHOLOGICAL PROFILE:
- Archetypes: {[a.value for a in profile.get('dominant_archetypes', [])]}
- Core complex: {profile.get('primary_complex', PsychologicalComplex(name='x', core_affect='x')).name}
- Core wound affect: {profile.get('primary_complex', PsychologicalComplex(name='x', core_affect='x')).core_affect}

The image should be strange, condensed (multiple meanings), emotionally charged.
Use dream logic - things can transform, locations shift, time is fluid.

THE DREAM BEGINS:"""

        content = await self._call_llm(prompt, depth=4, temperature=1.0)

        return AdvancedDialogue(
            speaker="THE DREAM IMAGE",
            speaker_type="dream",
            content=content,
            dream_mode=DreamLogicMode.SYMBOLIZATION,
            authenticity_level=0.9,
        )

    async def _process_dream_mode(
        self,
        persona_name: str,
        profile: Dict[str, Any],
        topic: str,
        mode: DreamLogicMode,
        previous_dialogues: List[AdvancedDialogue],
    ) -> AdvancedDialogue:
        """Process dream through specific mode."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:300]}..." for d in previous_dialogues])

        mode_instructions = {
            DreamLogicMode.CONDENSATION: "Multiple meanings collapse into single images. What are all the things this image represents?",
            DreamLogicMode.DISPLACEMENT: "The emotion has attached to the wrong object. What is the feeling really about?",
            DreamLogicMode.SYMBOLIZATION: "Abstract becomes concrete. What does each symbol stand for?",
            DreamLogicMode.AMPLIFICATION: "Expand the symbol through cultural and mythological associations. What does it connect to across human history?",
            DreamLogicMode.ACTIVE_IMAGINATION: "Dialogue with the dream figures. Let them speak and interact.",
        }

        prompt = f"""DREAM PROCESSING: {mode.value.upper()}

The dream so far:
{previous}

PROCESSING MODE: {mode.value}
{mode_instructions.get(mode, '')}

Continue the dream using {mode.value} logic.
For {persona_name}, regarding {topic}.

THE DREAM CONTINUES ({mode.value}):"""

        content = await self._call_llm(prompt, depth=4, temperature=1.0)

        return AdvancedDialogue(
            speaker=f"DREAM ({mode.value.upper()})",
            speaker_type="dream",
            content=content,
            dream_mode=mode,
            authenticity_level=0.9,
        )

    async def _interpret_dream(
        self,
        persona_name: str,
        profile: Dict[str, Any],
        dialogues: List[AdvancedDialogue],
    ) -> AdvancedDialogue:
        """Interpret the complete dream."""
        dream_content = "\n\n".join([d.content for d in dialogues if d.speaker_type == "dream"])

        prompt = f"""DREAM INTERPRETATION for {persona_name}

THE COMPLETE DREAM:
{dream_content}

PSYCHOLOGICAL PROFILE:
- Archetypes: {[a.value for a in profile.get('dominant_archetypes', [])]}
- Shadow: {[s.value for s in profile.get('shadow_archetypes', [])]}
- Core complex: {profile.get('primary_complex', PsychologicalComplex(name='x', core_affect='x')).name}

Interpret this dream from a depth psychological perspective.
What is the unconscious trying to communicate?
What compensatory message does the dream carry?
What integration is being sought?

THE INTERPRETATION:"""

        content = await self._call_llm(prompt, depth=3, temperature=0.7)

        return AdvancedDialogue(
            speaker="THE INTERPRETATION",
            speaker_type="interpretation",
            content=content,
            authenticity_level=0.8,
        )

    async def _apply_intervention(
        self,
        session: AdvancedDreamspaceSession,
        intervention: Intervention,
        profile: Dict[str, Any],
        intel_context: str,
    ):
        """Apply a therapeutic intervention."""
        previous = "\n".join([f"[{d.speaker}]: {d.content[:200]}..." for d in session.dialogues[-3:]])

        prompt = f"""THERAPEUTIC INTERVENTION in {session.persona_name}'s Dreamspace

Previous content:
{previous}

INTERVENTION TYPE: {intervention.intervention_type.value}
INTERVENTION: {intervention.content}
TARGET: {intervention.target}
INTENDED EFFECT: {intervention.intended_effect}

The Dreamspace responds to this intervention.
What shifts? What emerges? What resistance arises?

RESPONSE TO INTERVENTION:"""

        content = await self._call_llm(prompt, depth=session.depth_level)

        dialogue = AdvancedDialogue(
            speaker=f"RESPONSE TO {intervention.intervention_type.value.upper()}",
            speaker_type="intervention_response",
            content=content,
        )
        session.dialogues.append(dialogue)
        session.interventions.append(intervention)

    async def _synthesize_session(self, session: AdvancedDreamspaceSession):
        """Generate final synthesis of the session."""
        all_content = "\n\n".join([
            f"[{d.speaker}]: {d.content[:300]}..."
            for d in session.dialogues
        ])

        prompt = f"""INTEGRATION SYNTHESIS for {session.persona_name}

SESSION TYPE: {session.session_type}
DEPTH REACHED: {session.depth_level}
TOPIC: {session.topic}

FULL CONTENT:
{all_content}

ACTIVATED:
- Archetypes: {[a.value for a in session.active_archetypes]}
- Shadows: {[s.value for s in session.shadow_manifestations]}
- Defenses: {[d.value for d in session.defense_patterns]}

Provide psychological integration:
1. Integration narrative (what has been synthesized, 3-4 sentences)
2. Key insights (5-7 bullet points of deepest revelations)
3. Unresolved tensions (what remains in conflict)
4. Transformation edges (where growth/change is possible)

Format as JSON:
{{"integration_narrative": "...", "key_insights": ["..."], "unresolved_tensions": ["..."], "transformation_edges": ["..."]}}"""

        try:
            response = await self._call_llm(prompt, depth=3, temperature=0.4)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                session.integration_narrative = parsed.get("integration_narrative", "")
                session.key_insights = parsed.get("key_insights", [])
                session.unresolved_tensions = parsed.get("unresolved_tensions", [])
                session.transformation_edges = parsed.get("transformation_edges", [])
                session.integration_achieved = bool(session.integration_narrative)
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

    async def _call_llm(
        self,
        prompt: str,
        depth: int = 3,
        temperature: float = 0.9,
        max_tokens: int = 1000,
    ) -> str:
        """Call LLM with depth-appropriate model."""
        model = self.MODELS_BY_DEPTH.get(depth, "x-ai/grok-3-beta")

        session = await self._get_session()

        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error = await response.text()
                    logger.error(f"LLM error: {response.status} - {error}")
                    return f"[Error: {response.status}]"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[Error: {e}]"

    def _generate_default_advanced_profile(self, persona_name: str) -> Dict[str, Any]:
        """Generate default psychological profile."""
        return {
            "dominant_archetypes": [JungianArchetype.HERO, JungianArchetype.SAGE],
            "shadow_archetypes": [ShadowArchetype.TRICKSTER],
            "primary_complex": PsychologicalComplex(
                name="Achievement Complex",
                core_affect="drivenness",
                defense_mechanisms=[DefenseMechanism.INTELLECTUALIZATION],
            ),
            "ifs_parts": [
                IFSPart(
                    part_type=IFSPartType.MANAGER,
                    name="The Controller",
                    core_belief="I must maintain control",
                ),
                IFSPart(
                    part_type=IFSPartType.EXILE,
                    name="The Wounded Child",
                    core_belief="I am not enough",
                    age_frozen=8,
                ),
            ],
            "core_dialectics": [],
            "somatic_map": {},
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

async def advanced_dreamspace(
    persona_name: str,
    topic: str,
    mode: str = "depth",  # "depth", "confrontation", "dream"
    persona_id: Optional[str] = None,
    depth_level: int = 4,
    other_personas: Optional[List[Tuple[str, str]]] = None,
) -> AdvancedDreamspaceSession:
    """
    Run an advanced Dreamspace session.

    Usage:
        # Full depth exploration
        session = await advanced_dreamspace("Elon Musk", "power and legacy", depth_level=5)

        # Multi-persona confrontation
        session = await advanced_dreamspace(
            "Sam Altman",
            "AGI safety",
            mode="confrontation",
            other_personas=[("ilya_sutskever", "Ilya Sutskever")]
        )

        # Dream logic session
        session = await advanced_dreamspace("Ilya Sutskever", "betrayal", mode="dream")
    """
    engine = AdvancedDreamspaceEngine()
    pid = persona_id or persona_name.lower().replace(" ", "_")

    try:
        if mode == "confrontation" and other_personas:
            personas = [(pid, persona_name)] + other_personas
            return await engine.run_multipersona_confrontation(
                personas=personas,
                topic=topic,
                depth_level=depth_level,
            )
        elif mode == "dream":
            return await engine.run_dream_logic_session(
                persona_id=pid,
                persona_name=persona_name,
                topic=topic,
            )
        else:
            return await engine.run_depth_session(
                persona_id=pid,
                persona_name=persona_name,
                topic=topic,
                depth_level=depth_level,
            )
    finally:
        await engine.close()
