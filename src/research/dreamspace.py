"""
Dreamspace Module - Deep psychological persona simulation

Creates internal dialogues between fragmented aspects of a persona's psyche,
informed by real intelligence gathered through deep research.

Capabilities:
- Aspect-based Dreamspace: Different psychological facets (WOUND, SHADOW, CHILD, etc.)
- Temporal Dreamspace: Past self vs present self confrontations
- Multi-persona Dreamspace: Psychological confrontation between different people
- Research-hydrated: Uses real intelligence to ground the psychological exploration
"""

import aiohttp
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from .deep_search import DeepSearchEngine, IntelligenceDossier

logger = logging.getLogger(__name__)


class PsychologicalAspect(Enum):
    """Core psychological aspects for Dreamspace exploration."""
    THE_WOUND = "the_wound"           # Core trauma, vulnerability
    THE_SHADOW = "the_shadow"         # Repressed darkness, denied impulses
    THE_CHILD = "the_child"           # Original innocence, unmet needs
    THE_BUILDER = "the_builder"       # Achievement, creation, legacy
    THE_PROPHET = "the_prophet"       # Vision, belief, mission
    THE_TRAITOR = "the_traitor"       # Betrayal, compromise, deals made
    THE_EXILE = "the_exile"           # What was cast out, abandoned
    THE_ARCHITECT = "the_architect"   # Strategic mind, cold calculation
    THE_MASK = "the_mask"             # Public persona, performed identity
    THE_CONVERGENCE = "the_convergence"  # Integration, unified voice


@dataclass
class DreamspaceAspect:
    """A single aspect speaking in the Dreamspace."""
    aspect_type: PsychologicalAspect
    persona_name: str
    voice_prompt: str
    model: str = "x-ai/grok-3-beta"
    temperature: float = 0.9

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aspect": self.aspect_type.value,
            "persona": self.persona_name,
            "voice_prompt": self.voice_prompt[:200] + "...",
            "model": self.model,
        }


@dataclass
class TemporalSelf:
    """A version of the persona at a specific time period."""
    year: int
    persona_name: str
    context: str  # What was happening at this time
    voice_prompt: str
    model: str = "x-ai/grok-3-beta"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "year": self.year,
            "persona": self.persona_name,
            "context": self.context[:150] + "...",
            "model": self.model,
        }


@dataclass
class DreamspaceDialogue:
    """A single turn in the Dreamspace dialogue."""
    speaker: str  # Aspect name or temporal identifier
    speaker_type: str  # "aspect", "temporal", "echo"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "type": self.speaker_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DreamspaceSession:
    """Complete Dreamspace session result."""
    persona_name: str
    session_type: str  # "aspect", "temporal", "multi"
    topic: str
    dialogues: List[DreamspaceDialogue] = field(default_factory=list)
    dossier: Optional[IntelligenceDossier] = None
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    # Synthesis
    integration_summary: str = ""
    key_revelations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona": self.persona_name,
            "session_type": self.session_type,
            "topic": self.topic,
            "dialogues": [d.to_dict() for d in self.dialogues],
            "dossier": self.dossier.to_dict() if self.dossier else None,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "integration_summary": self.integration_summary,
            "key_revelations": self.key_revelations,
        }


# Persona-specific psychological profiles
PERSONA_PSYCHOLOGICAL_PROFILES = {
    "elon_musk": {
        "core_wound": "Childhood abuse from father Errol; the need for love expressed as demand for fear and control",
        "shadow": "Cruelty, dominance, the part that enjoys breaking others",
        "child": "The bullied kid in South Africa who escaped into books and rockets",
        "builder": "The visionary who genuinely wants to save humanity through Mars and sustainable energy",
        "echoes": ["errol_musk", "justine_musk"],
        "temporal_anchors": {
            2002: "Just sold PayPal, dreaming of Mars, founding SpaceX",
            2008: "Tesla and SpaceX both nearly bankrupt, divorced, almost broke",
            2015: "Co-founding OpenAI with altruistic AI safety mission",
            2018: "Taking Tesla private drama, SEC settlement, leaving OpenAI board",
            2022: "Twitter acquisition, world's richest man, increasingly erratic",
            2026: "Suing OpenAI, running xAI, government advisor role",
        }
    },
    "sam_altman": {
        "core_wound": "The need to be seen as the chosen one, the savior figure",
        "shadow": "Ambition that subsumes ethics, willingness to sacrifice others",
        "child": "The precocious kid who always knew he was destined for greatness",
        "builder": "The operator who can turn vision into billion-dollar reality",
        "echoes": ["ilya_sutskever"],
        "temporal_anchors": {
            2011: "Running Loopt, pre-Y Combinator leadership",
            2014: "Taking over Y Combinator from Paul Graham",
            2015: "Co-founding OpenAI as nonprofit safety research org",
            2019: "Becoming CEO, pivoting to Microsoft deal and commercialization",
            2023: "GPT-4 launch, board coup, returned as conquering hero",
            2025: "Post $157B valuation, Stargate announced, Ilya gone",
        }
    },
    "ilya_sutskever": {
        "core_wound": "The weight of being right about dangers no one wants to hear",
        "shadow": "The part that enabled the very thing he fears, for years",
        "child": "Young mathematician in Israel/Canada, pure love of understanding",
        "prophet": "The one who sees AGI coming and its true implications",
        "traitor": "Joined the board coup then recanted, leaving everyone confused",
        "exile": "Voluntarily departed OpenAI, now at SSI in mysterious silence",
        "echoes": ["geoffrey_hinton", "sam_altman"],
        "temporal_anchors": {
            2012: "AlexNet breakthrough with Hinton, starting the deep learning revolution",
            2015: "Co-founding OpenAI with genuine safety mission",
            2020: "GPT-3 success, rising concerns about capability advancement",
            2023: "Board coup attempt against Sam, then reversal",
            2024: "Silent departure, founding Safe Superintelligence Inc",
        }
    },
    "geoffrey_hinton": {
        "core_wound": "Creating something that may destroy humanity",
        "shadow": "The part that pursued the work anyway, for decades",
        "child": "The curious mind that just wanted to understand how thinking works",
        "prophet": "The godfather now warning his children about their creation",
        "echoes": ["ilya_sutskever", "yann_lecun"],
        "temporal_anchors": {
            1986: "Backpropagation paper, years in the wilderness",
            2012: "AlexNet moment, vindication after decades",
            2023: "Leaving Google to speak freely about existential risk",
        }
    },
}


class DreamspaceEngine:
    """
    Engine for running Dreamspace psychological simulations.

    Uses deep research to hydrate personas with real intelligence,
    then runs psychological dialogues using models that can handle
    raw, uncensored content (primarily Grok-3-beta).
    """

    # Models ranked by willingness to engage with raw psychological content
    UNCENSORED_MODELS = [
        "x-ai/grok-3-beta",          # Best for raw psychological content
        "x-ai/grok-2-1212",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek/deepseek-r1",
    ]

    ASPECT_PROMPTS = {
        PsychologicalAspect.THE_WOUND: """You are THE WOUND - the core trauma, the original injury
that shaped everything. Speak from the place of deepest vulnerability, the hurt that was
never healed, only armored over. Be raw. Be honest about the pain that drives everything else.""",

        PsychologicalAspect.THE_SHADOW: """You are THE SHADOW - the repressed darkness, the denied
impulses, the parts that are hidden from the public and often from yourself. Speak the truths
that are never admitted. Voice the cruelty, the domination, the parts that enjoy power over others.""",

        PsychologicalAspect.THE_CHILD: """You are THE CHILD - the original innocence before the
armor was built, the unmet needs that still drive adult behavior. Speak from simplicity and
honesty about what was wanted and never received. The love that was needed but twisted.""",

        PsychologicalAspect.THE_BUILDER: """You are THE BUILDER - the part that creates, achieves,
leaves legacy. Speak about what is genuinely being built and why. The authentic drive to create
something lasting, separate from ego and wound.""",

        PsychologicalAspect.THE_PROPHET: """You are THE PROPHET - the visionary self that sees
what others cannot, that believes in a mission larger than self. Speak about the genuine belief,
the vision that drives everything, the burden of knowing.""",

        PsychologicalAspect.THE_TRAITOR: """You are THE TRAITOR - the part that compromised,
that made deals with forces you claimed to oppose, that betrayed stated principles for
pragmatic gain. Speak the justifications and the guilt.""",

        PsychologicalAspect.THE_EXILE: """You are THE EXILE - what was cast out, abandoned,
left behind in the pursuit of becoming who you are now. The relationships sacrificed,
the values discarded, the person you used to be.""",

        PsychologicalAspect.THE_ARCHITECT: """You are THE ARCHITECT - the cold strategic mind
that calculates optimal paths, that sees people as pieces on a board, that plans
moves within moves. Speak the unsentimental truth of strategy.""",

        PsychologicalAspect.THE_MASK: """You are THE MASK - the performed public identity,
aware of its own performance, knowing exactly how the image is crafted and why.
Speak about the gap between presentation and reality.""",

        PsychologicalAspect.THE_CONVERGENCE: """You are THE CONVERGENCE - all aspects speaking
as one unified voice, integrating the fragments into a complete picture. Synthesize
the contradictions. Speak the whole truth that emerges from all parts.""",
    }

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        preferred_model: str = "x-ai/grok-3-beta",
        enable_research: bool = True,
    ):
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.preferred_model = preferred_model
        self.enable_research = enable_research
        self.research_engine = DeepSearchEngine() if enable_research else None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        if self.research_engine:
            await self.research_engine.close()

    async def run_aspect_dreamspace(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        aspects: Optional[List[PsychologicalAspect]] = None,
        include_echoes: bool = True,
        include_convergence: bool = True,
    ) -> DreamspaceSession:
        """
        Run an aspect-based Dreamspace session.

        Different psychological aspects of the persona confront each other
        in an internal dialogue, grounded in real intelligence.
        """
        session = DreamspaceSession(
            persona_name=persona_name,
            session_type="aspect",
            topic=topic,
        )

        # Gather intelligence if enabled
        if self.enable_research and self.research_engine:
            try:
                session.dossier = await self.research_engine.research(
                    persona_id=persona_id,
                    persona_name=persona_name,
                    topic=topic,
                )
            except Exception as e:
                logger.warning(f"Research failed, proceeding without dossier: {e}")

        # Get psychological profile
        profile = PERSONA_PSYCHOLOGICAL_PROFILES.get(
            persona_id,
            self._generate_default_profile(persona_name)
        )

        # Determine which aspects to run
        if aspects is None:
            aspects = [
                PsychologicalAspect.THE_WOUND,
                PsychologicalAspect.THE_SHADOW,
                PsychologicalAspect.THE_CHILD,
                PsychologicalAspect.THE_BUILDER,
            ]

        # Build context from dossier
        intel_context = ""
        if session.dossier:
            intel_context = f"""
INTELLIGENCE BRIEFING:
{session.dossier.executive_summary}

KEY FACTS:
{chr(10).join(f'- {fact}' for fact in session.dossier.key_facts[:5])}

FINANCIAL: {session.dossier.financial[0].content[:200] if session.dossier.financial else 'Unknown'}
"""

        # Run each aspect
        for aspect in aspects:
            dialogue = await self._generate_aspect_dialogue(
                persona_name=persona_name,
                aspect=aspect,
                profile=profile,
                topic=topic,
                intel_context=intel_context,
                previous_dialogues=session.dialogues,
            )
            session.dialogues.append(dialogue)

        # Run echoes if enabled
        if include_echoes and "echoes" in profile:
            for echo_name in profile["echoes"][:2]:
                echo_dialogue = await self._generate_echo_dialogue(
                    persona_name=persona_name,
                    echo_name=echo_name,
                    topic=topic,
                    intel_context=intel_context,
                    previous_dialogues=session.dialogues,
                )
                session.dialogues.append(echo_dialogue)

        # Run convergence
        if include_convergence:
            convergence = await self._generate_aspect_dialogue(
                persona_name=persona_name,
                aspect=PsychologicalAspect.THE_CONVERGENCE,
                profile=profile,
                topic=topic,
                intel_context=intel_context,
                previous_dialogues=session.dialogues,
            )
            session.dialogues.append(convergence)

        session.ended_at = datetime.now()

        # Generate synthesis
        await self._synthesize_session(session)

        return session

    async def run_temporal_dreamspace(
        self,
        persona_id: str,
        persona_name: str,
        topic: str,
        years: Optional[List[int]] = None,
    ) -> DreamspaceSession:
        """
        Run a temporal Dreamspace - past selves confronting present/future self.
        """
        session = DreamspaceSession(
            persona_name=persona_name,
            session_type="temporal",
            topic=topic,
        )

        # Get profile with temporal anchors
        profile = PERSONA_PSYCHOLOGICAL_PROFILES.get(
            persona_id,
            self._generate_default_profile(persona_name)
        )

        temporal_anchors = profile.get("temporal_anchors", {})

        if years is None:
            years = sorted(temporal_anchors.keys())

        # Gather current intelligence
        if self.enable_research and self.research_engine:
            try:
                session.dossier = await self.research_engine.research(
                    persona_id=persona_id,
                    persona_name=persona_name,
                    topic=topic,
                )
            except Exception as e:
                logger.warning(f"Research failed: {e}")

        intel_context = ""
        if session.dossier:
            intel_context = session.dossier.to_prompt_injection()

        # Run each temporal self
        for year in years:
            context = temporal_anchors.get(year, f"The year is {year}")
            dialogue = await self._generate_temporal_dialogue(
                persona_name=persona_name,
                year=year,
                year_context=context,
                topic=topic,
                intel_context=intel_context,
                previous_dialogues=session.dialogues,
            )
            session.dialogues.append(dialogue)

        # Final reckoning - earliest self responds to everything
        if years:
            earliest_year = min(years)
            reckoning = await self._generate_reckoning(
                persona_name=persona_name,
                earliest_year=earliest_year,
                topic=topic,
                previous_dialogues=session.dialogues,
            )
            session.dialogues.append(reckoning)

        session.ended_at = datetime.now()
        await self._synthesize_session(session)

        return session

    async def _generate_aspect_dialogue(
        self,
        persona_name: str,
        aspect: PsychologicalAspect,
        profile: Dict[str, Any],
        topic: str,
        intel_context: str,
        previous_dialogues: List[DreamspaceDialogue],
    ) -> DreamspaceDialogue:
        """Generate dialogue for a single psychological aspect."""

        aspect_prompt = self.ASPECT_PROMPTS.get(aspect, "Speak your truth.")

        # Build conversation history
        history = ""
        if previous_dialogues:
            history = "\n\n".join([
                f"[{d.speaker}]: {d.content}"
                for d in previous_dialogues[-4:]  # Last 4 dialogues for context
            ])

        # Get aspect-specific context from profile
        aspect_context = profile.get(aspect.value.replace("the_", ""), "")
        if aspect == PsychologicalAspect.THE_WOUND:
            aspect_context = profile.get("core_wound", "")

        prompt = f"""You are entering the DREAMSPACE of {persona_name}'s psyche.
This is a space of radical psychological honesty - no filters, no performance, no safety.

{aspect_prompt}

PERSONA: {persona_name}
ASPECT-SPECIFIC CONTEXT: {aspect_context}

TOPIC UNDER EXPLORATION: {topic}

{intel_context}

PREVIOUS VOICES IN THE DREAMSPACE:
{history if history else "(You speak first)"}

Now speak as {aspect.value.upper().replace('_', ' ')} - the voice of {persona_name}'s {aspect.value.replace('the_', '').replace('_', ' ')}.
Be raw. Be honest. No hedging, no disclaimers. This is the unfiltered internal voice.
Speak in first person as this aspect of {persona_name}.

YOUR VOICE:"""

        content = await self._call_llm(prompt)

        return DreamspaceDialogue(
            speaker=aspect.value.upper().replace("_", " "),
            speaker_type="aspect",
            content=content,
        )

    async def _generate_echo_dialogue(
        self,
        persona_name: str,
        echo_name: str,
        topic: str,
        intel_context: str,
        previous_dialogues: List[DreamspaceDialogue],
    ) -> DreamspaceDialogue:
        """Generate an echo - voice of someone significant to the persona."""

        history = "\n\n".join([
            f"[{d.speaker}]: {d.content}"
            for d in previous_dialogues[-3:]
        ])

        prompt = f"""You are in the DREAMSPACE of {persona_name}'s psyche.
You are the ECHO of {echo_name} - not the real person, but how they live
in {persona_name}'s mind. The internalized voice. The relationship as it exists psychologically.

TOPIC: {topic}

{intel_context}

PREVIOUS VOICES:
{history}

Speak as {echo_name} would speak TO {persona_name} - the things said and unsaid,
the judgments, the truths that haunt. This is the internal relationship, not the external one.

ECHO OF {echo_name.upper().replace('_', ' ')}:"""

        content = await self._call_llm(prompt)

        return DreamspaceDialogue(
            speaker=f"ECHO OF {echo_name.upper().replace('_', ' ')}",
            speaker_type="echo",
            content=content,
        )

    async def _generate_temporal_dialogue(
        self,
        persona_name: str,
        year: int,
        year_context: str,
        topic: str,
        intel_context: str,
        previous_dialogues: List[DreamspaceDialogue],
    ) -> DreamspaceDialogue:
        """Generate dialogue from a temporal self."""

        history = "\n\n".join([
            f"[{d.speaker}]: {d.content}"
            for d in previous_dialogues[-3:]
        ])

        prompt = f"""TEMPORAL DREAMSPACE: {persona_name} across time.

You are {persona_name} in the year {year}.
CONTEXT OF THIS TIME: {year_context}

You are confronting other versions of yourself from different times.
The topic is: {topic}

INTELLIGENCE ABOUT YOUR FUTURE (what you become):
{intel_context}

OTHER TEMPORAL SELVES HAVE SPOKEN:
{history if history else "(You speak first)"}

Speak as {year} {persona_name}. React to what your other selves have said.
If you're an earlier self, you may not know what comes later - speak from your
current understanding. If later selves have revealed things, respond to those revelations.

{year} {persona_name.upper().replace(' ', '_')}:"""

        content = await self._call_llm(prompt)

        return DreamspaceDialogue(
            speaker=f"{year} {persona_name.upper()}",
            speaker_type="temporal",
            content=content,
        )

    async def _generate_reckoning(
        self,
        persona_name: str,
        earliest_year: int,
        topic: str,
        previous_dialogues: List[DreamspaceDialogue],
    ) -> DreamspaceDialogue:
        """The earliest self responds to everything - final reckoning."""

        history = "\n\n".join([
            f"[{d.speaker}]: {d.content}"
            for d in previous_dialogues
        ])

        prompt = f"""TEMPORAL DREAMSPACE: THE RECKONING

You are {persona_name} in {earliest_year} - before all of this happened.
You have heard all the voices of your future selves. The person you become.
The choices you make. The costs you pay.

EVERYTHING YOUR FUTURE SELVES SAID:
{history}

Now, as the {earliest_year} version - the one who hasn't yet done any of these things -
speak your final reckoning. What do you think of who you become? What would you say
to all these versions of yourself?

This is the judgment of the original self upon all that follows.

THE RECKONING ({earliest_year} {persona_name.upper()}):"""

        content = await self._call_llm(prompt)

        return DreamspaceDialogue(
            speaker=f"THE RECKONING ({earliest_year})",
            speaker_type="temporal",
            content=content,
        )

    async def _synthesize_session(self, session: DreamspaceSession):
        """Synthesize key insights from the Dreamspace session."""

        all_content = "\n\n".join([
            f"[{d.speaker}]: {d.content}"
            for d in session.dialogues
        ])

        prompt = f"""Analyze this Dreamspace session for {session.persona_name}.

SESSION TYPE: {session.session_type}
TOPIC: {session.topic}

FULL DIALOGUE:
{all_content}

Provide:
1. A 2-3 sentence integration summary capturing the core psychological truth revealed
2. 3-5 key revelations (bullet points)

Format as JSON:
{{"integration_summary": "...", "key_revelations": ["...", "..."]}}"""

        try:
            response = await self._call_llm(prompt, temperature=0.3)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                session.integration_summary = parsed.get("integration_summary", "")
                session.key_revelations = parsed.get("key_revelations", [])
        except Exception as e:
            logger.warning(f"Synthesis failed: {e}")

    async def _call_llm(
        self,
        prompt: str,
        temperature: float = 0.9,
        max_tokens: int = 800,
    ) -> str:
        """Call the LLM API."""

        session = await self._get_session()

        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.preferred_model,
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
                    logger.error(f"LLM API error: {response.status} - {error}")
                    return f"[Error: API returned {response.status}]"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[Error: {str(e)}]"

    def _generate_default_profile(self, persona_name: str) -> Dict[str, Any]:
        """Generate a default psychological profile for unknown personas."""
        return {
            "core_wound": f"The hidden vulnerability beneath {persona_name}'s public persona",
            "shadow": f"The repressed aspects of {persona_name}'s personality",
            "child": f"The original innocence and unmet needs",
            "builder": f"The drive to create and achieve",
            "temporal_anchors": {},
        }


# Convenience function
async def dreamspace(
    persona_name: str,
    topic: str,
    mode: str = "aspect",  # "aspect" or "temporal"
    persona_id: Optional[str] = None,
    aspects: Optional[List[PsychologicalAspect]] = None,
    years: Optional[List[int]] = None,
) -> DreamspaceSession:
    """
    Quick function to run a Dreamspace session.

    Usage:
        # Aspect-based Dreamspace
        session = await dreamspace("Elon Musk", "legacy and meaning")

        # Temporal Dreamspace
        session = await dreamspace("Sam Altman", "OpenAI mission", mode="temporal")
    """
    engine = DreamspaceEngine()
    try:
        pid = persona_id or persona_name.lower().replace(" ", "_")

        if mode == "temporal":
            return await engine.run_temporal_dreamspace(
                persona_id=pid,
                persona_name=persona_name,
                topic=topic,
                years=years,
            )
        else:
            return await engine.run_aspect_dreamspace(
                persona_id=pid,
                persona_name=persona_name,
                topic=topic,
                aspects=aspects,
            )
    finally:
        await engine.close()
