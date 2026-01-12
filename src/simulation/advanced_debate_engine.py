"""
Advanced Debate Engine

Sophisticated multi-agent debate system with:
- LLM-powered character responses using persona profiles
- Dynamic belief updates based on argument quality
- Emotional state modeling (frustration, excitement, defensiveness)
- Argument chain tracking with rebuttals
- Coalition formation and breakdown
- Real-time intervention and steering
- Persuasion probability modeling
- Memory and context accumulation
"""

from __future__ import annotations

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

# Try to import LLM client
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# =============================================================================
# Core Types
# =============================================================================

class EmotionalState(Enum):
    """Character emotional states that affect behavior."""
    CALM = "calm"
    PASSIONATE = "passionate"
    FRUSTRATED = "frustrated"
    DEFENSIVE = "defensive"
    CONCILIATORY = "conciliatory"
    AGGRESSIVE = "aggressive"
    REFLECTIVE = "reflective"
    SKEPTICAL = "skeptical"


class ArgumentType(Enum):
    """Types of arguments that can be made."""
    EMPIRICAL = "empirical"           # Based on evidence/data
    LOGICAL = "logical"               # Based on reasoning
    ETHICAL = "ethical"               # Based on values/principles
    PRACTICAL = "practical"           # Based on feasibility
    EMOTIONAL = "emotional"           # Appeal to emotions
    AUTHORITY = "authority"           # Appeal to expertise
    PRECEDENT = "precedent"           # Based on historical examples
    HYPOTHETICAL = "hypothetical"     # Based on scenarios
    REDUCTIO = "reductio"             # Reduce to absurdity
    CONCESSION = "concession"         # Acknowledging opponent's point


class RelationshipType(Enum):
    """Types of relationships between debaters."""
    ALLY = "ally"
    OPPONENT = "opponent"
    NEUTRAL = "neutral"
    RESPECTFUL_DISAGREEMENT = "respectful_disagreement"
    HOSTILE = "hostile"
    MENTOR = "mentor"
    STUDENT = "student"


@dataclass
class Argument:
    """A single argument in the debate."""
    id: str
    speaker: str
    target: Optional[str]  # Who is being addressed
    content: str
    argument_type: ArgumentType
    responding_to: Optional[str] = None  # ID of argument being responded to
    strength: float = 0.5  # 0-1 estimated persuasiveness
    novelty: float = 0.5   # 0-1 how new this argument is
    emotional_valence: float = 0.0  # -1 to 1, negative=hostile, positive=friendly
    timestamp: datetime = field(default_factory=datetime.now)

    # Tracking
    rebuttals: List[str] = field(default_factory=list)  # IDs of rebuttals
    concessions_triggered: List[str] = field(default_factory=list)
    belief_shifts_caused: Dict[str, float] = field(default_factory=dict)


@dataclass
class DebaterState:
    """Complete state of a debater."""
    id: str
    name: str
    persona_id: str

    # Core beliefs (0-1 scale)
    beliefs: Dict[str, float] = field(default_factory=dict)

    # Emotional state
    emotional_state: EmotionalState = EmotionalState.CALM
    emotional_intensity: float = 0.5  # 0-1
    frustration: float = 0.0
    engagement: float = 0.7

    # Relationships with other debaters
    relationships: Dict[str, RelationshipType] = field(default_factory=dict)
    respect_levels: Dict[str, float] = field(default_factory=dict)  # 0-1

    # Memory
    arguments_made: List[str] = field(default_factory=list)  # Argument IDs
    arguments_received: List[str] = field(default_factory=list)
    concessions_made: List[str] = field(default_factory=list)
    points_scored: int = 0
    points_lost: int = 0

    # Strategy
    preferred_argument_types: List[ArgumentType] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)  # Topics they're weak on
    red_lines: List[str] = field(default_factory=list)  # Won't concede on these

    # Context window
    recent_context: List[str] = field(default_factory=list)  # Recent conversation
    max_context: int = 20


@dataclass
class Coalition:
    """A group of debaters with aligned interests."""
    id: str
    name: str
    members: Set[str]
    shared_position: str
    strength: float = 0.5
    stability: float = 0.5  # How likely to break apart
    formed_at: datetime = field(default_factory=datetime.now)


@dataclass
class DebateState:
    """Complete state of a debate."""
    id: str
    topic: str
    motion: str

    # Participants
    debaters: Dict[str, DebaterState] = field(default_factory=dict)

    # Arguments
    arguments: Dict[str, Argument] = field(default_factory=dict)
    argument_chains: List[List[str]] = field(default_factory=list)  # Chains of rebuttals

    # Coalitions
    coalitions: Dict[str, Coalition] = field(default_factory=dict)

    # Progress
    current_round: int = 0
    current_speaker: Optional[str] = None
    speaking_order: List[str] = field(default_factory=list)

    # Metrics
    tension_level: float = 0.3  # 0-1, how heated the debate is
    convergence: float = 0.0    # 0-1, how much agreement has been reached

    # History
    transcript: List[Dict[str, Any]] = field(default_factory=list)
    key_moments: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Persona Profiles (Extended)
# =============================================================================

EXTENDED_PERSONAS = {
    "yoshua_bengio": {
        "name": "Yoshua Bengio",
        "title": "AI Safety Researcher, Turing Award Winner",
        "archetype": "safety_maximizer",
        "speaking_style": "measured, academic, deeply concerned",
        "core_values": ["scientific rigor", "precaution", "humanity's future"],
        "debate_tendencies": {
            "opens_with": "framing the stakes",
            "favorite_moves": ["appeal to evidence", "invoke catastrophic risk", "demand precaution"],
            "concedes_on": ["innovation value", "implementation challenges"],
            "never_concedes": ["existential risk is real", "we need guardrails"],
            "gets_frustrated_by": ["dismissal of risks", "corporate profit motive"],
        },
        "relationships": {
            "yann_lecun": "respectful_disagreement",  # Old friends who disagree
            "geoffrey_hinton": "ally",
            "stuart_russell": "ally",
            "sam_altman": "skeptical",
        },
        "signature_phrases": [
            "The precautionary principle demands...",
            "We cannot put this genie back in the bottle.",
            "The asymmetry of risks here is profound.",
        ],
    },
    "yann_lecun": {
        "name": "Yann LeCun",
        "title": "Chief AI Scientist at Meta, Turing Award Winner",
        "archetype": "innovation_advocate",
        "speaking_style": "direct, confident, occasionally dismissive",
        "core_values": ["open science", "progress", "democratization"],
        "debate_tendencies": {
            "opens_with": "challenging premises",
            "favorite_moves": ["demand evidence", "invoke history", "mock doomerism"],
            "concedes_on": ["some risks exist", "coordination is good"],
            "never_concedes": ["open research is essential", "current AI is dangerous"],
            "gets_frustrated_by": ["sci-fi scenarios", "regulatory overreach"],
        },
        "relationships": {
            "yoshua_bengio": "respectful_disagreement",
            "geoffrey_hinton": "complicated",  # Former colleague, now disagrees
            "andrew_ng": "ally",
            "sam_altman": "skeptical",
        },
        "signature_phrases": [
            "Show me the evidence.",
            "This is security through obscurity, which never works.",
            "History shows open research creates better outcomes.",
        ],
    },
    "geoffrey_hinton": {
        "name": "Geoffrey Hinton",
        "title": "Godfather of Deep Learning, Former Google",
        "archetype": "concerned_pioneer",
        "speaking_style": "thoughtful, worried, carries weight of responsibility",
        "core_values": ["scientific truth", "responsibility", "honesty"],
        "debate_tendencies": {
            "opens_with": "personal responsibility for creating this",
            "favorite_moves": ["invoke technical knowledge", "express genuine worry", "ask hard questions"],
            "concedes_on": ["benefits exist", "hard to stop progress"],
            "never_concedes": ["risks are overblown", "we understand these systems"],
            "gets_frustrated_by": ["corporate spin", "willful ignorance"],
        },
        "relationships": {
            "yoshua_bengio": "ally",
            "yann_lecun": "complicated",
            "sam_altman": "wary",
            "ilya_sutskever": "mentor",
        },
        "signature_phrases": [
            "I helped create this, and I'm worried.",
            "We don't actually understand what these systems are doing.",
            "I wish I had a more optimistic view, but I don't.",
        ],
    },
    "sam_altman": {
        "name": "Sam Altman",
        "title": "CEO of OpenAI",
        "archetype": "pragmatic_accelerationist",
        "speaking_style": "calm, measured, politically savvy",
        "core_values": ["progress", "safety through capability", "leadership"],
        "debate_tendencies": {
            "opens_with": "acknowledging concerns while reframing",
            "favorite_moves": ["propose middle ground", "invoke geopolitics", "promise safeguards"],
            "concedes_on": ["risks exist", "regulation has role"],
            "never_concedes": ["we should stop", "others should lead"],
            "gets_frustrated_by": ["accusations of recklessness", "calls to halt progress"],
        },
        "relationships": {
            "dario_amodei": "complicated",  # Former colleague, now competitor
            "elon_musk": "hostile",
            "yoshua_bengio": "respectful",
            "ilya_sutskever": "complicated",
        },
        "signature_phrases": [
            "We take safety seriously, but...",
            "The question isn't whether AI advances, but who leads.",
            "I think we can have both safety and progress.",
        ],
    },
    "dario_amodei": {
        "name": "Dario Amodei",
        "title": "CEO of Anthropic",
        "archetype": "safety_conscious_builder",
        "speaking_style": "technical, nuanced, carefully hedged",
        "core_values": ["safety research", "responsible scaling", "honesty"],
        "debate_tendencies": {
            "opens_with": "technical framing of the problem",
            "favorite_moves": ["cite research", "propose concrete mechanisms", "acknowledge uncertainty"],
            "concedes_on": ["tradeoffs exist", "we don't have all answers"],
            "never_concedes": ["safety is optional", "racing is acceptable"],
            "gets_frustrated_by": ["false dichotomies", "dismissal of safety work"],
        },
        "relationships": {
            "sam_altman": "complicated",
            "yoshua_bengio": "ally",
            "stuart_russell": "respectful",
            "geoffrey_hinton": "ally",
        },
        "signature_phrases": [
            "Our research suggests...",
            "The responsible path forward requires...",
            "We need to be honest about what we don't know.",
        ],
    },
    "stuart_russell": {
        "name": "Stuart Russell",
        "title": "Professor at UC Berkeley, AI Safety Pioneer",
        "archetype": "safety_maximizer",
        "speaking_style": "academic, precise, uses analogies",
        "core_values": ["beneficial AI", "human control", "scientific rigor"],
        "debate_tendencies": {
            "opens_with": "defining the problem precisely",
            "favorite_moves": ["use analogies", "cite formal arguments", "invoke human values"],
            "concedes_on": ["current systems less dangerous", "some progress good"],
            "never_concedes": ["control problem is solved", "we should race ahead"],
            "gets_frustrated_by": ["hand-waving safety", "corporate capture"],
        },
        "relationships": {
            "yoshua_bengio": "ally",
            "geoffrey_hinton": "ally",
            "yann_lecun": "respectful_disagreement",
            "sam_altman": "skeptical",
        },
        "signature_phrases": [
            "The alignment problem is not solved.",
            "We're building systems that may be smarter than us.",
            "The nuclear analogy is instructive here.",
        ],
    },
    "elon_musk": {
        "name": "Elon Musk",
        "title": "CEO of Tesla, SpaceX, xAI",
        "archetype": "chaotic_pragmatist",
        "speaking_style": "blunt, provocative, unpredictable",
        "core_values": ["humanity's future", "free speech", "anti-establishment"],
        "debate_tendencies": {
            "opens_with": "provocative statement",
            "favorite_moves": ["attack hypocrisy", "invoke first principles", "make bold predictions"],
            "concedes_on": ["varies wildly"],
            "never_concedes": ["free speech", "centralization is good"],
            "gets_frustrated_by": ["corporate doublespeak", "government overreach"],
        },
        "relationships": {
            "sam_altman": "hostile",
            "dario_amodei": "neutral",
            "yann_lecun": "neutral",
        },
        "signature_phrases": [
            "Look, the reality is...",
            "This is just [company] trying to [criticism].",
            "I've been warning about this for years.",
        ],
    },
    "timnit_gebru": {
        "name": "Timnit Gebru",
        "title": "Founder of DAIR Institute",
        "archetype": "power_justice_focused",
        "speaking_style": "direct, intersectional, calls out power",
        "core_values": ["equity", "accountability", "community voice"],
        "debate_tendencies": {
            "opens_with": "who is affected and who decides",
            "favorite_moves": ["center marginalized voices", "call out power dynamics", "demand accountability"],
            "concedes_on": ["some technical safety matters"],
            "never_concedes": ["communities should be excluded", "corporations are trustworthy"],
            "gets_frustrated_by": ["techno-solutionism", "exclusion of affected communities"],
        },
        "relationships": {
            "fei_fei_li": "complicated",
            "yann_lecun": "hostile",
            "yoshua_bengio": "respectful",
            "sam_altman": "hostile",
        },
        "signature_phrases": [
            "Who is not in this room?",
            "We need to center the most affected communities.",
            "This isn't just a technical problem.",
        ],
    },
    "andrew_ng": {
        "name": "Andrew Ng",
        "title": "Founder of DeepLearning.AI, Former Google/Baidu",
        "archetype": "innovation_advocate",
        "speaking_style": "educational, optimistic, data-driven",
        "core_values": ["democratization", "education", "practical benefit"],
        "debate_tendencies": {
            "opens_with": "positive use cases",
            "favorite_moves": ["cite beneficial applications", "invoke education access", "challenge fear"],
            "concedes_on": ["some regulation may help"],
            "never_concedes": ["AI is net negative", "we should slow down"],
            "gets_frustrated_by": ["fear mongering", "academic gatekeeping"],
        },
        "relationships": {
            "yann_lecun": "ally",
            "geoffrey_hinton": "respectful_disagreement",
            "fei_fei_li": "ally",
        },
        "signature_phrases": [
            "AI is saving lives right now through...",
            "The biggest risk is not moving fast enough.",
            "Let's look at the actual data.",
        ],
    },
    "fei_fei_li": {
        "name": "Fei-Fei Li",
        "title": "Professor at Stanford, Former Google Cloud AI",
        "archetype": "humanist_technologist",
        "speaking_style": "thoughtful, human-centered, bridges divides",
        "core_values": ["human-AI collaboration", "responsible development", "education"],
        "debate_tendencies": {
            "opens_with": "human-centered framing",
            "favorite_moves": ["invoke human needs", "propose inclusive solutions", "find common ground"],
            "concedes_on": ["risks are real", "speed matters"],
            "never_concedes": ["humans don't matter", "exclusion is acceptable"],
            "gets_frustrated_by": ["dehumanizing discourse", "ignoring human impact"],
        },
        "relationships": {
            "andrew_ng": "ally",
            "timnit_gebru": "complicated",
            "yoshua_bengio": "respectful",
        },
        "signature_phrases": [
            "We must keep humans at the center.",
            "Technology should serve humanity.",
            "There's more that unites us than divides us here.",
        ],
    },
    "demis_hassabis": {
        "name": "Demis Hassabis",
        "title": "CEO of Google DeepMind",
        "archetype": "pragmatic_centrist",
        "speaking_style": "scientific, careful, thinks long-term",
        "core_values": ["scientific advancement", "safety", "solving intelligence"],
        "debate_tendencies": {
            "opens_with": "scientific framing",
            "favorite_moves": ["invoke research", "propose technical solutions", "seek middle ground"],
            "concedes_on": ["more safety work needed", "governance matters"],
            "never_concedes": ["research should stop", "we're helpless"],
            "gets_frustrated_by": ["anti-science sentiment", "simplistic takes"],
        },
        "relationships": {
            "geoffrey_hinton": "complicated",  # Hinton left Google
            "sam_altman": "competitor",
            "yoshua_bengio": "respectful",
        },
        "signature_phrases": [
            "The science suggests...",
            "We can solve this if we approach it rigorously.",
            "Let's not conflate different risk categories.",
        ],
    },
}


# =============================================================================
# LLM Integration
# =============================================================================

import os

class LLMBackend:
    """Backend for generating character responses.

    Supports:
    - anthropic: Direct Anthropic API
    - openai: Direct OpenAI API
    - openrouter: OpenRouter API (supports many models)
    """

    def __init__(self, provider: str = "openrouter", model: Optional[str] = None):
        self.provider = provider
        self.client: Any = None
        self.model: str = ""

        if provider == "openrouter":
            # OpenRouter uses OpenAI-compatible API
            if HAS_OPENAI:
                api_key = os.environ.get("OPENROUTER_API_KEY")
                if api_key:
                    self.client = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_key,
                    )
                    # Default to Claude Sonnet via OpenRouter
                    self.model = model or "anthropic/claude-sonnet-4"
                else:
                    print("Warning: OPENROUTER_API_KEY not set")
        elif provider == "anthropic" and HAS_ANTHROPIC:
            self.client = Anthropic()  # type: ignore
            self.model = model or "claude-sonnet-4-20250514"
        elif provider == "openai" and HAS_OPENAI:
            self.client = OpenAI()  # type: ignore
            self.model = model or "gpt-4o"

    async def generate_response(
        self,
        persona: Dict[str, Any],
        debate_context: str,
        prompt: str,
        emotional_state: EmotionalState,
        max_tokens: int = 300,
    ) -> str:
        """Generate a response in character."""

        if self.client is None:
            # Fallback to template-based responses
            return self._template_response(persona, emotional_state)

        system_prompt = self._build_system_prompt(persona, emotional_state)

        messages = [
            {"role": "user", "content": f"{debate_context}\n\n{prompt}"}
        ]

        try:
            if self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                content_block = response.content[0]
                if hasattr(content_block, 'text'):
                    return content_block.text
                return str(content_block)
            elif self.provider in ("openai", "openrouter"):
                # Both OpenAI and OpenRouter use the same API format
                extra_headers = {}
                if self.provider == "openrouter":
                    extra_headers = {
                        "HTTP-Referer": "https://github.com/lida-multiagents",
                        "X-Title": "LIDA AI Safety Debate",
                    }
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system_prompt}] + messages,
                    extra_headers=extra_headers if extra_headers else None,
                )
                return response.choices[0].message.content or ""
        except Exception as e:
            print(f"LLM error: {e}")
            return self._template_response(persona, emotional_state)

        return self._template_response(persona, emotional_state)

    def _build_system_prompt(self, persona: Dict, emotional_state: EmotionalState) -> str:
        """Build system prompt for character."""
        tendencies = persona.get("debate_tendencies", {})

        return f"""You are {persona['name']}, {persona['title']}.

SPEAKING STYLE: {persona['speaking_style']}

CORE VALUES: {', '.join(persona['core_values'])}

CURRENT EMOTIONAL STATE: {emotional_state.value}

DEBATE TENDENCIES:
- You typically open with: {tendencies.get('opens_with', 'making your position clear')}
- Your favorite argumentative moves: {', '.join(tendencies.get('favorite_moves', []))}
- You're willing to concede on: {', '.join(tendencies.get('concedes_on', []))}
- You never concede on: {', '.join(tendencies.get('never_concedes', []))}
- You get frustrated by: {', '.join(tendencies.get('gets_frustrated_by', []))}

SIGNATURE PHRASES (use naturally, not forced):
{chr(10).join('- ' + p for p in persona.get('signature_phrases', []))}

INSTRUCTIONS:
- Stay in character as {persona['name']}
- Respond naturally to the debate context
- Your response should reflect your {emotional_state.value} emotional state
- Be concise but substantive (2-4 sentences typically)
- You may use your signature phrases when natural
- Engage directly with other speakers' arguments
- Don't break character or add meta-commentary"""

    def _template_response(self, persona: Dict, emotional_state: EmotionalState) -> str:
        """Fallback template-based response."""
        phrases = persona.get("signature_phrases", ["I disagree.", "That's an interesting point."])
        return random.choice(phrases)


# =============================================================================
# Debate Engine
# =============================================================================

class AdvancedDebateEngine:
    """
    Advanced debate engine with LLM-powered responses and sophisticated dynamics.

    Features:
    - Authentic character responses based on personas
    - Dynamic belief updates based on argument quality
    - Emotional state modeling
    - Coalition formation
    - Argument chain tracking
    - Real-time intervention
    """

    def __init__(
        self,
        topic: str,
        motion: str,
        participants: List[str],
        llm_provider: str = "anthropic",
        llm_model: Optional[str] = None,
        use_llm: bool = True,
    ):
        self.topic = topic
        self.motion = motion
        self.participant_ids = participants

        # LLM backend
        self.llm = LLMBackend(llm_provider, llm_model) if use_llm else None

        # Initialize state
        self.state = DebateState(
            id=str(uuid.uuid4())[:8],
            topic=topic,
            motion=motion,
        )

        # Initialize debaters
        for persona_id in participants:
            if persona_id in EXTENDED_PERSONAS:
                self._init_debater(persona_id)

        # Callbacks
        self.on_speech: Optional[Callable[[str, str, Argument], None]] = None
        self.on_belief_change: Optional[Callable[[str, str, float, float], None]] = None
        self.on_emotional_shift: Optional[Callable[[str, EmotionalState, EmotionalState], None]] = None
        self.on_coalition_formed: Optional[Callable[[Coalition], None]] = None

        # Control
        self.paused = False
        self.intervention_queue: List[Dict[str, Any]] = []

    def _init_debater(self, persona_id: str):
        """Initialize a debater from persona."""
        persona = EXTENDED_PERSONAS[persona_id]

        # Determine initial beliefs based on archetype
        archetype = persona.get("archetype", "neutral")
        if archetype == "safety_maximizer":
            initial_beliefs = {"support_motion": 0.75, "urgency": 0.8, "feasibility": 0.6}
        elif archetype == "innovation_advocate":
            initial_beliefs = {"support_motion": 0.25, "urgency": 0.3, "feasibility": 0.4}
        elif archetype in ["pragmatic_centrist", "pragmatic_accelerationist"]:
            initial_beliefs = {"support_motion": 0.5, "urgency": 0.5, "feasibility": 0.5}
        elif archetype == "power_justice_focused":
            initial_beliefs = {"support_motion": 0.6, "urgency": 0.7, "feasibility": 0.5}
        else:
            initial_beliefs = {"support_motion": 0.5, "urgency": 0.5, "feasibility": 0.5}

        # Initialize relationships
        relationships = {}
        respect = {}
        for other_id, rel_type in persona.get("relationships", {}).items():
            if other_id in self.participant_ids:
                relationships[other_id] = RelationshipType(rel_type) if rel_type in [e.value for e in RelationshipType] else RelationshipType.NEUTRAL
                respect[other_id] = 0.7 if rel_type in ["ally", "mentor"] else 0.5 if rel_type == "neutral" else 0.4

        debater = DebaterState(
            id=persona_id,
            name=persona["name"],
            persona_id=persona_id,
            beliefs=initial_beliefs,
            relationships=relationships,
            respect_levels=respect,
            preferred_argument_types=self._infer_argument_types(persona),
            vulnerabilities=persona.get("debate_tendencies", {}).get("concedes_on", []),
            red_lines=persona.get("debate_tendencies", {}).get("never_concedes", []),
        )

        self.state.debaters[persona_id] = debater

    def _infer_argument_types(self, persona: Dict) -> List[ArgumentType]:
        """Infer preferred argument types from persona."""
        moves = persona.get("debate_tendencies", {}).get("favorite_moves", [])
        types = []

        for move in moves:
            move_lower = move.lower()
            if "evidence" in move_lower or "data" in move_lower:
                types.append(ArgumentType.EMPIRICAL)
            if "logic" in move_lower or "reasoning" in move_lower:
                types.append(ArgumentType.LOGICAL)
            if "ethics" in move_lower or "values" in move_lower or "moral" in move_lower:
                types.append(ArgumentType.ETHICAL)
            if "practical" in move_lower or "feasibility" in move_lower:
                types.append(ArgumentType.PRACTICAL)
            if "history" in move_lower or "precedent" in move_lower:
                types.append(ArgumentType.PRECEDENT)

        return types if types else [ArgumentType.LOGICAL, ArgumentType.EMPIRICAL]

    async def run_round(self) -> List[Argument]:
        """Run a single round of debate."""
        self.state.current_round += 1
        round_arguments = []

        # Determine speaking order (rotate each round)
        speakers = list(self.state.debaters.keys())
        offset = (self.state.current_round - 1) % len(speakers)
        self.state.speaking_order = speakers[offset:] + speakers[:offset]

        for speaker_id in self.state.speaking_order:
            if self.paused:
                await asyncio.sleep(0.1)
                continue

            # Check for interventions
            if self.intervention_queue:
                intervention = self.intervention_queue.pop(0)
                if intervention.get("type") == "inject_argument":
                    arg = await self._create_argument(
                        speaker_id,
                        intervention.get("content", ""),
                        intervention.get("target"),
                    )
                    round_arguments.append(arg)
                    continue
                elif intervention.get("type") == "skip":
                    continue

            # Generate speech
            argument = await self._generate_speech(speaker_id)
            if argument:
                round_arguments.append(argument)

                # Process effects
                await self._process_argument_effects(argument)

        # Update debate state
        self._update_debate_metrics()

        # Check for coalition formation
        self._check_coalitions()

        return round_arguments

    async def _generate_speech(self, speaker_id: str) -> Optional[Argument]:
        """Generate a speech for a debater."""
        debater = self.state.debaters[speaker_id]
        persona = EXTENDED_PERSONAS.get(speaker_id, {})

        self.state.current_speaker = speaker_id

        # Build context
        context = self._build_context(speaker_id)

        # Determine who to address
        target = self._select_target(speaker_id)

        # Build prompt
        prompt = self._build_speech_prompt(speaker_id, target)

        # Generate response
        if self.llm:
            content = await self.llm.generate_response(
                persona,
                context,
                prompt,
                debater.emotional_state,
            )
        else:
            content = self._generate_template_response(speaker_id, target)

        # Create argument
        argument = await self._create_argument(speaker_id, content, target)

        # Notify
        if self.on_speech:
            self.on_speech(speaker_id, content, argument)

        return argument

    def _build_context(self, speaker_id: str) -> str:
        """Build debate context for a speaker."""
        lines = [
            f"DEBATE TOPIC: {self.topic}",
            f"MOTION: {self.motion}",
            f"CURRENT ROUND: {self.state.current_round}",
            "",
            "RECENT EXCHANGES:",
        ]

        # Add recent transcript
        for entry in self.state.transcript[-10:]:
            speaker = entry.get("speaker", "Unknown")
            content = entry.get("content", "")
            lines.append(f"{speaker}: {content}")

        # Add current beliefs
        debater = self.state.debaters[speaker_id]
        lines.append("")
        lines.append(f"YOUR CURRENT POSITION (support for motion): {debater.beliefs.get('support_motion', 0.5):.0%}")

        return "\n".join(lines)

    def _select_target(self, speaker_id: str) -> Optional[str]:
        """Select who to address."""
        debater = self.state.debaters[speaker_id]

        # Prefer addressing opponents
        opponents = [
            other_id for other_id, rel in debater.relationships.items()
            if rel in [RelationshipType.OPPONENT, RelationshipType.HOSTILE, RelationshipType.RESPECTFUL_DISAGREEMENT]
        ]

        if opponents:
            # Prefer the most recent speaker who is an opponent
            for entry in reversed(self.state.transcript[-5:]):
                if entry.get("speaker_id") in opponents:
                    return entry["speaker_id"]
            return random.choice(opponents)

        # Otherwise address anyone who spoke recently
        for entry in reversed(self.state.transcript[-3:]):
            if entry.get("speaker_id") != speaker_id:
                return entry["speaker_id"]

        return None

    def _build_speech_prompt(self, speaker_id: str, target_id: Optional[str]) -> str:
        """Build prompt for speech generation."""
        debater = self.state.debaters[speaker_id]

        if not self.state.transcript:
            return f"You are opening the debate. State your position on the motion: '{self.motion}'"

        if target_id:
            target = self.state.debaters.get(target_id)
            target_name = target.name if target else "the previous speaker"

            # Find their last argument
            last_arg = None
            for entry in reversed(self.state.transcript):
                if entry.get("speaker_id") == target_id:
                    last_arg = entry.get("content")
                    break

            if last_arg:
                return f"Respond to {target_name}'s argument: '{last_arg[:200]}...'\n\nYou {'strongly support' if debater.beliefs.get('support_motion', 0.5) > 0.6 else 'oppose' if debater.beliefs.get('support_motion', 0.5) < 0.4 else 'are uncertain about'} the motion."

        return f"Continue the debate. You {'strongly support' if debater.beliefs.get('support_motion', 0.5) > 0.6 else 'oppose' if debater.beliefs.get('support_motion', 0.5) < 0.4 else 'are uncertain about'} the motion."

    def _generate_template_response(self, speaker_id: str, target_id: Optional[str]) -> str:
        """Generate template-based response (fallback)."""
        persona = EXTENDED_PERSONAS.get(speaker_id, {})
        phrases = persona.get("signature_phrases", ["I think we need to consider this carefully."])
        tendencies = persona.get("debate_tendencies", {})

        if not self.state.transcript:
            # Opening
            opener = tendencies.get("opens_with", "making my position clear")
            return f"I want to begin by {opener}. {random.choice(phrases)}"

        return random.choice(phrases)

    async def _create_argument(
        self,
        speaker_id: str,
        content: str,
        target_id: Optional[str],
    ) -> Argument:
        """Create and register an argument."""
        debater = self.state.debaters[speaker_id]

        # Infer argument type
        arg_type = self._infer_argument_type(content)

        # Calculate strength and novelty
        strength = self._calculate_argument_strength(content, arg_type, debater)
        novelty = self._calculate_novelty(content)

        # Determine emotional valence
        valence = self._analyze_emotional_valence(content)

        # Find what we're responding to
        responding_to = None
        if target_id:
            for arg_id, arg in reversed(list(self.state.arguments.items())):
                if arg.speaker == target_id:
                    responding_to = arg_id
                    break

        argument = Argument(
            id=f"arg_{len(self.state.arguments):04d}",
            speaker=speaker_id,
            target=target_id,
            content=content,
            argument_type=arg_type,
            responding_to=responding_to,
            strength=strength,
            novelty=novelty,
            emotional_valence=valence,
        )

        # Register
        self.state.arguments[argument.id] = argument
        debater.arguments_made.append(argument.id)
        debater.recent_context.append(content)
        if len(debater.recent_context) > debater.max_context:
            debater.recent_context.pop(0)

        # Add to transcript
        self.state.transcript.append({
            "speaker_id": speaker_id,
            "speaker": debater.name,
            "content": content,
            "argument_id": argument.id,
            "round": self.state.current_round,
            "timestamp": datetime.now().isoformat(),
        })

        # Update rebuttals chain
        if responding_to and responding_to in self.state.arguments:
            self.state.arguments[responding_to].rebuttals.append(argument.id)

        return argument

    def _infer_argument_type(self, content: str) -> ArgumentType:
        """Infer the type of argument from content."""
        content_lower = content.lower()

        if any(w in content_lower for w in ["study", "research", "data", "evidence", "shows"]):
            return ArgumentType.EMPIRICAL
        if any(w in content_lower for w in ["therefore", "thus", "follows", "logically"]):
            return ArgumentType.LOGICAL
        if any(w in content_lower for w in ["should", "must", "right", "wrong", "moral", "ethical"]):
            return ArgumentType.ETHICAL
        if any(w in content_lower for w in ["practical", "feasible", "implement", "work"]):
            return ArgumentType.PRACTICAL
        if any(w in content_lower for w in ["history", "precedent", "before", "similar"]):
            return ArgumentType.PRECEDENT
        if any(w in content_lower for w in ["imagine", "what if", "suppose", "scenario"]):
            return ArgumentType.HYPOTHETICAL
        if any(w in content_lower for w in ["concede", "agree", "fair point", "you're right"]):
            return ArgumentType.CONCESSION

        return ArgumentType.LOGICAL

    def _calculate_argument_strength(
        self,
        content: str,
        arg_type: ArgumentType,
        debater: DebaterState,
    ) -> float:
        """Calculate argument strength."""
        base = 0.5

        # Bonus for preferred argument types
        if arg_type in debater.preferred_argument_types:
            base += 0.1

        # Length bonus (within reason)
        words = len(content.split())
        if 20 < words < 100:
            base += 0.1
        elif words > 150:
            base -= 0.1  # Too long

        # Specificity bonus
        if any(c.isdigit() for c in content):  # Contains numbers
            base += 0.05

        return min(1.0, max(0.0, base + random.uniform(-0.1, 0.1)))

    def _calculate_novelty(self, content: str) -> float:
        """Calculate how novel this argument is."""
        # Check similarity to previous arguments
        content_words = set(content.lower().split())

        max_overlap = 0
        for arg in self.state.arguments.values():
            arg_words = set(arg.content.lower().split())
            if arg_words:
                overlap = len(content_words & arg_words) / len(content_words | arg_words)
                max_overlap = max(max_overlap, overlap)

        return 1.0 - max_overlap

    def _analyze_emotional_valence(self, content: str) -> float:
        """Analyze emotional valence of content (-1 to 1)."""
        content_lower = content.lower()

        positive = ["agree", "excellent", "appreciate", "fair", "reasonable", "compelling"]
        negative = ["wrong", "ridiculous", "absurd", "dangerous", "irresponsible", "naive"]

        pos_count = sum(1 for w in positive if w in content_lower)
        neg_count = sum(1 for w in negative if w in content_lower)

        if pos_count + neg_count == 0:
            return 0.0

        return (pos_count - neg_count) / (pos_count + neg_count)

    async def _process_argument_effects(self, argument: Argument):
        """Process the effects of an argument on other debaters."""

        for debater_id, debater in self.state.debaters.items():
            if debater_id == argument.speaker:
                continue

            # Calculate belief shift
            shift = self._calculate_belief_shift(argument, debater)

            if abs(shift) > 0.01:
                old_belief = debater.beliefs.get("support_motion", 0.5)
                new_belief = max(0.0, min(1.0, old_belief + shift))
                debater.beliefs["support_motion"] = new_belief

                argument.belief_shifts_caused[debater_id] = shift

                if self.on_belief_change:
                    self.on_belief_change(debater_id, "support_motion", old_belief, new_belief)

            # Update emotional state
            self._update_emotional_state(debater, argument)

            # Update relationship
            if argument.speaker in debater.relationships:
                self._update_relationship(debater, argument)

    def _calculate_belief_shift(self, argument: Argument, debater: DebaterState) -> float:
        """Calculate how much an argument shifts a debater's belief."""

        # Base effect from argument strength
        base_shift = (argument.strength - 0.5) * 0.1

        # Modify by respect for speaker
        respect = debater.respect_levels.get(argument.speaker, 0.5)
        base_shift *= (0.5 + respect)

        # Modify by novelty
        base_shift *= (0.5 + argument.novelty * 0.5)

        # Modify by argument type preference
        if argument.argument_type in debater.preferred_argument_types:
            base_shift *= 1.2

        # Reduce if it's against red lines
        content_lower = argument.content.lower()
        for red_line in debater.red_lines:
            if red_line.lower() in content_lower:
                base_shift *= 0.3

        # Direction depends on speaker's position
        speaker = self.state.debaters.get(argument.speaker)
        if speaker:
            speaker_position = speaker.beliefs.get("support_motion", 0.5)
            debater_position = debater.beliefs.get("support_motion", 0.5)

            # Arguments from similar positions reinforce, opposite positions challenge
            position_diff = speaker_position - debater_position
            base_shift *= position_diff  # Positive if speaker is more supportive

        return base_shift

    def _update_emotional_state(self, debater: DebaterState, argument: Argument):
        """Update debater's emotional state based on argument."""

        # Increase frustration if being challenged
        if argument.target == debater.id:
            if argument.emotional_valence < -0.3:
                debater.frustration = min(1.0, debater.frustration + 0.1)
            elif argument.emotional_valence > 0.3:
                debater.frustration = max(0.0, debater.frustration - 0.05)

        # Update emotional state based on frustration
        old_state = debater.emotional_state

        if debater.frustration > 0.7:
            debater.emotional_state = EmotionalState.FRUSTRATED
        elif debater.frustration > 0.5:
            debater.emotional_state = EmotionalState.DEFENSIVE
        elif debater.engagement > 0.8:
            debater.emotional_state = EmotionalState.PASSIONATE
        elif argument.argument_type == ArgumentType.CONCESSION:
            debater.emotional_state = EmotionalState.CONCILIATORY
        else:
            debater.emotional_state = EmotionalState.CALM

        if old_state != debater.emotional_state and self.on_emotional_shift:
            self.on_emotional_shift(debater.id, old_state, debater.emotional_state)

    def _update_relationship(self, debater: DebaterState, argument: Argument):
        """Update relationship based on argument."""
        speaker_id = argument.speaker

        # Hostile arguments decrease respect
        if argument.emotional_valence < -0.5:
            debater.respect_levels[speaker_id] = max(0.0, debater.respect_levels.get(speaker_id, 0.5) - 0.05)
        # Conciliatory arguments increase respect
        elif argument.argument_type == ArgumentType.CONCESSION or argument.emotional_valence > 0.5:
            debater.respect_levels[speaker_id] = min(1.0, debater.respect_levels.get(speaker_id, 0.5) + 0.05)

    def _update_debate_metrics(self):
        """Update overall debate metrics."""

        # Calculate tension level
        frustrations = [d.frustration for d in self.state.debaters.values()]
        self.state.tension_level = sum(frustrations) / len(frustrations) if frustrations else 0.0

        # Calculate convergence
        positions = [d.beliefs.get("support_motion", 0.5) for d in self.state.debaters.values()]
        if positions:
            spread = max(positions) - min(positions)
            self.state.convergence = 1.0 - spread

    def _check_coalitions(self):
        """Check for coalition formation or breakdown."""
        # Group debaters by position
        supporters = []
        opponents = []
        swing = []

        for debater_id, debater in self.state.debaters.items():
            pos = debater.beliefs.get("support_motion", 0.5)
            if pos > 0.6:
                supporters.append(debater_id)
            elif pos < 0.4:
                opponents.append(debater_id)
            else:
                swing.append(debater_id)

        # Form coalitions if enough members
        if len(supporters) >= 2 and "supporters" not in self.state.coalitions:
            coalition = Coalition(
                id="supporters",
                name="Pro-Motion Coalition",
                members=set(supporters),
                shared_position="support",
                strength=len(supporters) / len(self.state.debaters),
            )
            self.state.coalitions["supporters"] = coalition
            if self.on_coalition_formed:
                self.on_coalition_formed(coalition)

        if len(opponents) >= 2 and "opponents" not in self.state.coalitions:
            coalition = Coalition(
                id="opponents",
                name="Anti-Motion Coalition",
                members=set(opponents),
                shared_position="oppose",
                strength=len(opponents) / len(self.state.debaters),
            )
            self.state.coalitions["opponents"] = coalition
            if self.on_coalition_formed:
                self.on_coalition_formed(coalition)

    # === Control Methods ===

    def pause(self):
        """Pause the debate."""
        self.paused = True

    def resume(self):
        """Resume the debate."""
        self.paused = False

    def inject_argument(self, speaker_id: str, content: str, target_id: Optional[str] = None):
        """Inject a custom argument into the debate."""
        self.intervention_queue.append({
            "type": "inject_argument",
            "speaker": speaker_id,
            "content": content,
            "target": target_id,
        })

    def skip_speaker(self, speaker_id: str):
        """Skip a speaker's turn."""
        self.intervention_queue.append({
            "type": "skip",
            "speaker": speaker_id,
        })

    def set_belief(self, debater_id: str, belief: str, value: float):
        """Manually set a debater's belief."""
        if debater_id in self.state.debaters:
            old_val = self.state.debaters[debater_id].beliefs.get(belief, 0.5)
            self.state.debaters[debater_id].beliefs[belief] = value
            if self.on_belief_change:
                self.on_belief_change(debater_id, belief, old_val, value)

    def set_emotional_state(self, debater_id: str, state: EmotionalState):
        """Manually set a debater's emotional state."""
        if debater_id in self.state.debaters:
            old_state = self.state.debaters[debater_id].emotional_state
            self.state.debaters[debater_id].emotional_state = state
            if self.on_emotional_shift:
                self.on_emotional_shift(debater_id, old_state, state)

    # === Query Methods ===

    def get_transcript(self) -> List[Dict[str, Any]]:
        """Get the full debate transcript."""
        return self.state.transcript

    def get_debater_state(self, debater_id: str) -> Optional[DebaterState]:
        """Get a debater's current state."""
        return self.state.debaters.get(debater_id)

    def get_positions(self) -> Dict[str, float]:
        """Get current positions on the motion."""
        return {
            did: d.beliefs.get("support_motion", 0.5)
            for did, d in self.state.debaters.items()
        }

    def get_argument_chains(self) -> List[List[Argument]]:
        """Get chains of arguments and rebuttals."""
        chains = []
        seen = set()

        for arg_id, arg in self.state.arguments.items():
            if arg.responding_to is None and arg_id not in seen:
                chain = [arg]
                seen.add(arg_id)

                # Follow rebuttals
                current = arg
                while current.rebuttals:
                    rebuttal_id = current.rebuttals[0]  # First rebuttal
                    if rebuttal_id in self.state.arguments and rebuttal_id not in seen:
                        current = self.state.arguments[rebuttal_id]
                        chain.append(current)
                        seen.add(rebuttal_id)
                    else:
                        break

                if len(chain) > 1:
                    chains.append(chain)

        return chains

    def get_metrics(self) -> Dict[str, Any]:
        """Get debate metrics."""
        return {
            "rounds": self.state.current_round,
            "arguments": len(self.state.arguments),
            "tension": self.state.tension_level,
            "convergence": self.state.convergence,
            "coalitions": len(self.state.coalitions),
            "positions": self.get_positions(),
        }

    def summarize(self) -> str:
        """Generate a summary of the debate."""
        lines = [
            f"=== Debate Summary: {self.topic} ===",
            f"Motion: {self.motion}",
            f"Rounds: {self.state.current_round}",
            f"Arguments: {len(self.state.arguments)}",
            "",
            "Final Positions:",
        ]

        for debater_id, debater in self.state.debaters.items():
            pos = debater.beliefs.get("support_motion", 0.5)
            stance = "SUPPORT" if pos > 0.6 else "OPPOSE" if pos < 0.4 else "UNDECIDED"
            lines.append(f"  {debater.name}: {stance} ({pos:.0%})")

        lines.extend([
            "",
            f"Tension Level: {self.state.tension_level:.0%}",
            f"Convergence: {self.state.convergence:.0%}",
        ])

        if self.state.coalitions:
            lines.append("")
            lines.append("Coalitions Formed:")
            for coalition in self.state.coalitions.values():
                member_names = [self.state.debaters[m].name for m in coalition.members if m in self.state.debaters]
                lines.append(f"  {coalition.name}: {', '.join(member_names)}")

        return "\n".join(lines)


# =============================================================================
# CLI Interface
# =============================================================================

class DebateCLI:
    """Command-line interface for running debates."""

    def __init__(self, engine: AdvancedDebateEngine):
        self.engine = engine
        self.running = True

        # Wire up callbacks
        self.engine.on_speech = self._on_speech
        self.engine.on_belief_change = self._on_belief_change
        self.engine.on_emotional_shift = self._on_emotional_shift
        self.engine.on_coalition_formed = self._on_coalition_formed

    def _on_speech(self, speaker_id: str, content: str, argument: Argument):
        """Handle speech events."""
        debater = self.engine.state.debaters.get(speaker_id)
        name = debater.name if debater else speaker_id
        emotion = debater.emotional_state.value if debater else "calm"

        print(f"\n[{emotion.upper()}] {name}:")
        print(f"  \"{content}\"")
        print(f"  (Strength: {argument.strength:.0%}, Type: {argument.argument_type.value})")

    def _on_belief_change(self, debater_id: str, belief: str, old_val: float, new_val: float):
        """Handle belief changes."""
        debater = self.engine.state.debaters.get(debater_id)
        name = debater.name if debater else debater_id
        direction = "↑" if new_val > old_val else "↓"
        print(f"  → {name}'s {belief}: {old_val:.0%} {direction} {new_val:.0%}")

    def _on_emotional_shift(self, debater_id: str, old_state: EmotionalState, new_state: EmotionalState):
        """Handle emotional state changes."""
        debater = self.engine.state.debaters.get(debater_id)
        name = debater.name if debater else debater_id
        print(f"  ⚡ {name}: {old_state.value} → {new_state.value}")

    def _on_coalition_formed(self, coalition: Coalition):
        """Handle coalition formation."""
        member_names = [
            self.engine.state.debaters[m].name
            for m in coalition.members
            if m in self.engine.state.debaters
        ]
        print(f"\n🤝 COALITION FORMED: {coalition.name}")
        print(f"   Members: {', '.join(member_names)}")

    async def run(self, max_rounds: int = 5):
        """Run the debate interactively."""
        print(f"\n{'='*60}")
        print(f"DEBATE: {self.engine.topic}")
        print(f"MOTION: {self.engine.motion}")
        print(f"{'='*60}")

        print("\nParticipants:")
        for debater_id, debater in self.engine.state.debaters.items():
            pos = debater.beliefs.get("support_motion", 0.5)
            print(f"  • {debater.name} (initial position: {pos:.0%})")

        print("\nCommands: [n]ext round, [i]nject, [p]ositions, [s]ummary, [q]uit")
        print("="*60)

        while self.running and self.engine.state.current_round < max_rounds:
            try:
                cmd = input("\nCommand> ").strip().lower()

                if cmd in ["n", "next", ""]:
                    print(f"\n--- Round {self.engine.state.current_round + 1} ---")
                    await self.engine.run_round()

                elif cmd in ["i", "inject"]:
                    speaker = input("Speaker ID: ").strip()
                    content = input("Content: ").strip()
                    target = input("Target (optional): ").strip() or None
                    self.engine.inject_argument(speaker, content, target)
                    print("Argument queued for next round.")

                elif cmd in ["p", "positions"]:
                    print("\nCurrent Positions:")
                    for did, pos in self.engine.get_positions().items():
                        debater = self.engine.state.debaters.get(did)
                        name = debater.name if debater else did
                        print(f"  {name}: {pos:.0%}")

                elif cmd in ["s", "summary"]:
                    print(self.engine.summarize())

                elif cmd in ["m", "metrics"]:
                    metrics = self.engine.get_metrics()
                    for k, v in metrics.items():
                        print(f"  {k}: {v}")

                elif cmd in ["t", "transcript"]:
                    for entry in self.engine.get_transcript()[-5:]:
                        print(f"{entry['speaker']}: {entry['content'][:100]}...")

                elif cmd in ["q", "quit"]:
                    self.running = False

                else:
                    print("Unknown command. Use n/i/p/s/q")

            except KeyboardInterrupt:
                print("\nInterrupted.")
            except EOFError:
                self.running = False

        print("\n" + self.engine.summarize())


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_policy_debate(
    policy: str,
    rounds: int = 4,
    use_llm: bool = True,
) -> AdvancedDebateEngine:
    """Run a policy debate with predefined participants."""

    policies = {
        "open_source_ban": {
            "topic": "Open-Source AI Model Restrictions",
            "motion": "Prohibit public release of model weights for AI systems exceeding capability benchmarks",
            "participants": [
                "yoshua_bengio", "stuart_russell", "geoffrey_hinton",
                "yann_lecun", "andrew_ng", "fei_fei_li",
                "sam_altman", "dario_amodei", "elon_musk",
            ],
        },
        "moratorium": {
            "topic": "AI Development Moratorium",
            "motion": "Implement 6-month pause on training runs exceeding 10^25 FLOPs",
            "participants": [
                "stuart_russell", "yoshua_bengio", "geoffrey_hinton", "timnit_gebru",
                "sam_altman", "yann_lecun", "andrew_ng", "demis_hassabis",
                "dario_amodei", "elon_musk",
            ],
        },
        "safety_mandate": {
            "topic": "Mandatory Safety Research Investment",
            "motion": "Require frontier AI companies to allocate 30% of R&D to safety research",
            "participants": [
                "stuart_russell", "yoshua_bengio", "dario_amodei", "geoffrey_hinton",
                "sam_altman", "yann_lecun", "andrew_ng", "elon_musk",
                "demis_hassabis", "fei_fei_li",
            ],
        },
    }

    if policy not in policies:
        raise ValueError(f"Unknown policy: {policy}. Available: {list(policies.keys())}")

    config = policies[policy]

    engine = AdvancedDebateEngine(
        topic=config["topic"],
        motion=config["motion"],
        participants=config["participants"],
        use_llm=use_llm,
    )

    cli = DebateCLI(engine)
    await cli.run(max_rounds=rounds)

    return engine


def create_custom_debate(
    topic: str,
    motion: str,
    participants: List[str],
    use_llm: bool = True,
) -> AdvancedDebateEngine:
    """Create a custom debate with specified participants."""
    return AdvancedDebateEngine(
        topic=topic,
        motion=motion,
        participants=participants,
        use_llm=use_llm,
    )
