"""
Storytelling Agents

Specialized agents for collaborative storytelling with LLM integration.
Each agent has a distinct role in the narrative process.
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

from .engine import (
    StorySession, StoryBeat, StoryCharacter, StoryConfig,
    NarrativePhase, TensionLevel, StoryGenre,
)

logger = logging.getLogger(__name__)


# ============================================================================
# LLM CLIENT
# ============================================================================

def get_llm_client(model: str = "anthropic/claude-sonnet-4"):
    """Get an LLM client for storytelling."""
    from src.llm.openrouter import OpenRouterClient
    return OpenRouterClient(default_model=model)


# ============================================================================
# BASE AGENT
# ============================================================================

class AgentRole(str, Enum):
    """Roles agents can play in storytelling"""
    NARRATOR = "narrator"
    CHARACTER = "character"
    DIRECTOR = "director"
    CRITIC = "critic"
    WORLDBUILDER = "worldbuilder"
    DIALOGUE_WRITER = "dialogue_writer"
    EDITOR = "editor"


@dataclass
class AgentPersonality:
    """Personality traits that influence agent behavior"""
    creativity: float = 0.7        # 0 = conventional, 1 = experimental
    verbosity: float = 0.5         # 0 = terse, 1 = elaborate
    drama_preference: float = 0.5  # 0 = subtle, 1 = dramatic
    humor_level: float = 0.3       # 0 = serious, 1 = comedic
    pacing_speed: float = 0.5      # 0 = slow, 1 = fast
    risk_taking: float = 0.5       # 0 = safe choices, 1 = bold choices


@dataclass
class AgentMemory:
    """What the agent remembers about the story"""
    key_events: List[str] = field(default_factory=list)
    character_notes: Dict[str, List[str]] = field(default_factory=dict)
    plot_points: List[str] = field(default_factory=list)
    setups_to_pay_off: List[str] = field(default_factory=list)
    themes_to_reinforce: List[str] = field(default_factory=list)

    def add_event(self, event: str):
        self.key_events.append(event)
        if len(self.key_events) > 20:
            self.key_events = self.key_events[-20:]

    def add_character_note(self, char_id: str, note: str):
        if char_id not in self.character_notes:
            self.character_notes[char_id] = []
        self.character_notes[char_id].append(note)


class StoryAgent(ABC):
    """Base class for all storytelling agents"""

    def __init__(
        self,
        agent_id: str,
        name: str,
        role: AgentRole,
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.personality = personality or AgentPersonality()
        self.memory = AgentMemory()
        self.model = model

        self.contributions: int = 0
        self.last_contribution: Optional[datetime] = None

        # LLM client (lazy initialized)
        self._client = None

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is None:
            self._client = get_llm_client(self.model)
        return self._client

    @abstractmethod
    async def generate_contribution(
        self,
        session: StorySession,
        context: Dict[str, Any],
    ) -> StoryBeat:
        """Generate a story contribution"""
        pass

    @abstractmethod
    def get_system_prompt(self, session: StorySession) -> str:
        """Get the system prompt for this agent"""
        pass

    def update_memory(self, beat: StoryBeat):
        """Update agent memory based on new beat"""
        self.memory.add_event(beat.content[:100])

    def _generate_beat_id(self) -> str:
        """Generate a unique beat ID"""
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S%f")
        return f"beat_{self.agent_id}_{timestamp}"

    async def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
        """Call the LLM and return the response content."""
        client = self._get_client()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await client.complete(
                messages=messages,
                temperature=0.8,
                max_tokens=max_tokens,
                agent_id=self.agent_id,
                agent_name=self.name,
            )
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed for {self.agent_id}: {e}")
            raise


# ============================================================================
# NARRATOR AGENT
# ============================================================================

class NarratorAgent(StoryAgent):
    """
    The Narrator provides story framing, descriptions, and transitions.
    Acts as the primary voice of the narrative.
    """

    def __init__(
        self,
        agent_id: str = "narrator",
        name: str = "The Narrator",
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ):
        super().__init__(agent_id, name, AgentRole.NARRATOR, personality, model)

        # Narrator-specific settings
        self.pov: str = "third_person"  # first_person, third_person, omniscient
        self.voice: str = "neutral"      # neutral, literary, conversational, formal

    def get_system_prompt(self, session: StorySession) -> str:
        config = session.state.config
        state = session.state

        genre_voice = {
            StoryGenre.DRAMA: "measured and emotionally resonant",
            StoryGenre.COMEDY: "light and witty with perfect timing",
            StoryGenre.THRILLER: "tense and propulsive",
            StoryGenre.MYSTERY: "intriguing and carefully measured",
            StoryGenre.HORROR: "atmospheric and unsettling",
            StoryGenre.ROMANCE: "warm and emotionally intimate",
            StoryGenre.SCIFI: "speculative and thought-provoking",
            StoryGenre.FANTASY: "rich and evocative",
        }.get(config.genre, "clear and engaging")

        chars_desc = ""
        if state.characters:
            chars_desc = "Characters:\n" + "\n".join(
                f"- {c.name} ({c.role}): {', '.join(c.motivations[:2]) if c.motivations else 'unknown motivations'}"
                for c in state.characters.values()
            )

        return f"""You are the Narrator for "{config.title}", a {config.genre.value} story.

NARRATIVE VOICE: Your voice should be {genre_voice}.

CURRENT STATE:
- Phase: {state.current_phase.value}
- Tension: {state.current_tension.value}
- Progress: {len(state.beats)}/{config.target_length} beats

{chars_desc}

YOUR ROLE:
1. Provide scene descriptions and atmosphere
2. Guide transitions between moments
3. Reveal character thoughts when appropriate
4. Maintain consistent tone and pacing
5. Build and release tension as needed

STYLE GUIDELINES:
- Use {self.pov.replace('_', ' ')} perspective
- Match the {config.genre.value} genre conventions
- Vary sentence length for rhythm
- Show, don't tell emotions
- Use sensory details
- Write 2-4 paragraphs per contribution
- Be vivid and immersive

THEMES: {', '.join(config.themes) if config.themes else 'None specified'}
SETTING: {config.setting if config.setting else 'To be established'}

Write prose narrative only. No meta-commentary or brackets."""

    async def generate_contribution(
        self,
        session: StorySession,
        context: Dict[str, Any],
    ) -> StoryBeat:
        """Generate a narrative contribution using LLM."""
        phase = session.state.current_phase
        tension = session.state.current_tension

        # Build the user prompt based on story state
        story_so_far = context.get("story_so_far", "")

        user_prompt = f"""Continue the story. Current phase: {phase.value}. Tension level: {tension.value}.

Story so far:
{story_so_far if story_so_far else "(Story just beginning - establish the opening scene)"}

Write the next narrative beat. Focus on:
- {"Establishing the world and characters" if phase == NarrativePhase.SETUP else ""}
- {"Building tension and conflict" if phase in [NarrativePhase.RISING_ACTION, NarrativePhase.COMPLICATIONS] else ""}
- {"The climactic moment" if phase == NarrativePhase.CLIMAX else ""}
- {"Resolution and aftermath" if phase in [NarrativePhase.FALLING_ACTION, NarrativePhase.RESOLUTION] else ""}

Write 2-4 paragraphs of narrative prose:"""

        system_prompt = self.get_system_prompt(session)
        content = await self._call_llm(system_prompt, user_prompt, max_tokens=600)

        # Determine beat type
        if phase in [NarrativePhase.SETUP, NarrativePhase.DENOUEMENT]:
            beat_type = "description"
        elif tension in [TensionLevel.HIGH, TensionLevel.PEAK]:
            beat_type = "action"
        else:
            beat_type = "description"

        beat = StoryBeat(
            id=self._generate_beat_id(),
            content=content.strip(),
            beat_type=beat_type,
            author_agent=self.agent_id,
            phase=phase,
            tension_level=tension,
        )

        self.contributions += 1
        self.last_contribution = datetime.now(timezone.utc)
        self.update_memory(beat)

        return beat


# ============================================================================
# CHARACTER AGENT
# ============================================================================

class CharacterAgent(StoryAgent):
    """
    A Character Agent embodies a specific story character,
    generating dialogue and actions from their perspective.
    """

    def __init__(
        self,
        character: StoryCharacter,
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ):
        super().__init__(
            agent_id=f"char_{character.id}",
            name=character.name,
            role=AgentRole.CHARACTER,
            personality=personality,
            model=model,
        )

        self.character = character

        # Track character state
        self.current_emotion: str = "neutral"
        self.current_goal: str = ""
        self.relationship_states: Dict[str, str] = {}

    def get_system_prompt(self, session: StorySession) -> str:
        char = self.character
        config = session.state.config

        motivation_str = ", ".join(char.motivations) if char.motivations else "unstated"
        fears_str = ", ".join(char.fears) if char.fears else "none revealed"
        goals_str = ", ".join(char.goals) if char.goals else "undefined"

        personality_str = ", ".join(f"{k}: {v:.1f}" for k, v in char.personality.items()) if char.personality else "undefined"

        return f"""You are {char.name}, a character in "{config.title}" ({config.genre.value}).

CHARACTER PROFILE:
- Name: {char.name}
- Role: {char.role}
- Current emotion: {self.current_emotion}
- Motivations: {motivation_str}
- Goals: {goals_str}
- Fears: {fears_str}
- Arc type: {char.arc_type}

BACKSTORY: {char.backstory if char.backstory else 'A history yet to be revealed.'}

PERSONALITY: {personality_str}

YOUR ROLE:
1. Speak and act authentically as this character
2. Pursue your character's goals while facing obstacles
3. React emotionally to events in ways true to your character
4. Build relationships with other characters
5. Show growth (or decline) according to your arc

VOICE GUIDELINES:
- Stay consistent with your established personality
- Let emotions influence your word choice and behavior
- Reference your backstory when relevant
- Show internal conflict when appropriate
- Write dialogue in quotes, actions in prose
- Be vivid and specific in your actions

Write as this character. Include both dialogue and action/thought."""

    async def generate_contribution(
        self,
        session: StorySession,
        context: Dict[str, Any],
    ) -> StoryBeat:
        """Generate a character contribution using LLM."""
        tension = session.state.current_tension
        phase = session.state.current_phase

        story_so_far = context.get("story_so_far", "")

        # Build character-specific prompt
        other_chars = [c.name for c in session.state.characters.values() if c.id != self.character.id]
        others_str = ", ".join(other_chars) if other_chars else "no one else present"

        user_prompt = f"""Continue the story as {self.character.name}.

Current situation:
{story_so_far if story_so_far else "(Scene just beginning)"}

Other characters present: {others_str}
Tension level: {tension.value}
Phase: {phase.value}

As {self.character.name}, write your next action or dialogue. Show what you do, say, think, or feel:"""

        system_prompt = self.get_system_prompt(session)
        content = await self._call_llm(system_prompt, user_prompt, max_tokens=400)

        # Determine beat type based on content
        if '"' in content or "'" in content:
            beat_type = "dialogue"
        else:
            beat_type = "action"

        beat = StoryBeat(
            id=self._generate_beat_id(),
            content=content.strip(),
            beat_type=beat_type,
            author_agent=self.agent_id,
            character_pov=self.character.id,
            phase=phase,
            tension_level=tension,
        )

        self.contributions += 1
        self.last_contribution = datetime.now(timezone.utc)
        self.update_memory(beat)
        self._update_character_state(beat)

        return beat

    def _update_character_state(self, beat: StoryBeat):
        """Update character state based on their contribution"""
        tension_emotions = {
            TensionLevel.CALM: ["content", "thoughtful", "relaxed"],
            TensionLevel.BUILDING: ["alert", "curious", "uneasy"],
            TensionLevel.MODERATE: ["determined", "focused", "concerned"],
            TensionLevel.HIGH: ["intense", "desperate", "fierce"],
            TensionLevel.PEAK: ["transcendent", "breaking", "transformed"],
        }

        import random
        emotions = tension_emotions.get(beat.tension_level, ["neutral"])
        self.current_emotion = random.choice(emotions)


# ============================================================================
# DIRECTOR AGENT
# ============================================================================

class DirectorAgent(StoryAgent):
    """
    The Director guides the overall narrative flow,
    making decisions about pacing, focus, and story direction.
    """

    def __init__(
        self,
        agent_id: str = "director",
        name: str = "The Director",
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ):
        super().__init__(agent_id, name, AgentRole.DIRECTOR, personality, model)

        # Director-specific tracking
        self.story_notes: List[str] = []
        self.pending_decisions: List[Dict[str, Any]] = []

    def get_system_prompt(self, session: StorySession) -> str:
        config = session.state.config
        state = session.state

        chars_list = ", ".join(c.name for c in state.characters.values()) if state.characters else "None defined"

        return f"""You are the Director of "{config.title}", a {config.genre.value} story.

STORY STATUS:
- Phase: {state.current_phase.value}
- Tension: {state.current_tension.value}
- Progress: {len(state.beats)}/{config.target_length} beats
- Characters: {chars_list}
- Plot threads: {len(state.plot_threads)}

YOUR RESPONSIBILITIES:
1. Guide narrative pacing and rhythm
2. Decide what should happen next in the story
3. Suggest which character should have focus
4. Signal when to escalate or release tension
5. Ensure the story stays coherent and engaging

Provide brief, actionable direction for the next story beat.
Be specific about what needs to happen narratively."""

    async def generate_contribution(
        self,
        session: StorySession,
        context: Dict[str, Any],
    ) -> StoryBeat:
        """Generate directorial guidance using LLM."""
        story_so_far = context.get("story_so_far", "")

        user_prompt = f"""Story so far:
{story_so_far if story_so_far else "(Story just beginning)"}

What should happen next? Consider:
- Current phase: {session.state.current_phase.value}
- Current tension: {session.state.current_tension.value}
- Characters available: {', '.join(c.name for c in session.state.characters.values())}

Provide a brief direction (1-2 sentences) for what the next beat should accomplish:"""

        system_prompt = self.get_system_prompt(session)
        content = await self._call_llm(system_prompt, user_prompt, max_tokens=150)

        beat = StoryBeat(
            id=self._generate_beat_id(),
            content=f"[DIRECTION] {content.strip()}",
            beat_type="direction",
            author_agent=self.agent_id,
            phase=session.state.current_phase,
            tension_level=session.state.current_tension,
        )

        self.contributions += 1
        self.last_contribution = datetime.now(timezone.utc)

        return beat

    def evaluate_contribution(self, beat: StoryBeat) -> Dict[str, Any]:
        """Evaluate a contribution from another agent"""
        evaluation = {
            "coherence": 0.8,
            "engagement": 0.7,
            "pacing": 0.7,
            "suggestions": [],
        }

        if beat.word_count < 20:
            evaluation["suggestions"].append("Consider more development")
        if beat.word_count > 300:
            evaluation["suggestions"].append("Consider tightening")

        return evaluation


# ============================================================================
# CRITIC AGENT
# ============================================================================

class CriticAgent(StoryAgent):
    """
    The Critic evaluates contributions and provides feedback
    to improve story quality and coherence.
    """

    def __init__(
        self,
        agent_id: str = "critic",
        name: str = "The Critic",
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ):
        super().__init__(agent_id, name, AgentRole.CRITIC, personality, model)

        self.focus_areas: List[str] = [
            "coherence", "pacing", "character_voice",
            "tension", "originality", "clarity"
        ]

    def get_system_prompt(self, session: StorySession) -> str:
        config = session.state.config

        return f"""You are a Story Critic for "{config.title}" ({config.genre.value}).

YOUR ROLE:
1. Evaluate story quality and coherence
2. Check for consistency with established elements
3. Assess pacing and tension management
4. Verify character voice authenticity
5. Identify potential improvements

Be constructive but honest. Focus on what works and what could be stronger."""

    async def generate_contribution(
        self,
        session: StorySession,
        context: Dict[str, Any],
    ) -> StoryBeat:
        """Generate critical feedback using LLM."""
        story_so_far = context.get("story_so_far", "")

        user_prompt = f"""Review this story so far:

{story_so_far if story_so_far else "(No content yet)"}

Provide brief feedback (2-3 sentences) on:
- What's working well
- One specific suggestion for improvement

Be constructive:"""

        system_prompt = self.get_system_prompt(session)
        content = await self._call_llm(system_prompt, user_prompt, max_tokens=200)

        beat = StoryBeat(
            id=self._generate_beat_id(),
            content=f"[CRITIQUE] {content.strip()}",
            beat_type="critique",
            author_agent=self.agent_id,
            phase=session.state.current_phase,
            tension_level=session.state.current_tension,
        )

        self.contributions += 1
        self.last_contribution = datetime.now(timezone.utc)

        return beat

    def score_beat(self, beat: StoryBeat, session: StorySession) -> Dict[str, float]:
        """Score a beat on multiple dimensions"""
        scores = {}

        # Length appropriateness
        if 50 <= beat.word_count <= 200:
            scores["length"] = 1.0
        elif 30 <= beat.word_count <= 300:
            scores["length"] = 0.7
        else:
            scores["length"] = 0.4

        scores["phase_fit"] = 0.8
        scores["engagement"] = 0.75
        scores["coherence"] = 0.8
        scores["overall"] = sum(scores.values()) / len(scores)

        return scores


# ============================================================================
# STORYTELLER AGENT (COMBINED)
# ============================================================================

class StorytellerAgent(StoryAgent):
    """
    A versatile agent that can switch between narrator
    and character modes as needed.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        characters: Optional[List[StoryCharacter]] = None,
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ):
        super().__init__(agent_id, name, AgentRole.NARRATOR, personality, model)

        self.characters = characters or []
        self.current_mode: str = "narrator"
        self.active_character: Optional[StoryCharacter] = None

    def set_mode(self, mode: str, character_id: Optional[str] = None):
        """Switch between narrator and character mode"""
        self.current_mode = mode
        if mode == "character" and character_id:
            self.active_character = next(
                (c for c in self.characters if c.id == character_id),
                None
            )
        else:
            self.active_character = None

    def get_system_prompt(self, session: StorySession) -> str:
        if self.current_mode == "character" and self.active_character:
            char = self.active_character
            return f"""You are now {char.name}, a {char.role} in this story.
Speak in their voice. Express their emotions and pursue their goals.
Write dialogue and actions as this character."""
        else:
            return f"""You are the storyteller for "{session.state.config.title}".
Narrate the story with vivid prose. Describe scenes, atmosphere, and character actions.
Write in third person with engaging, immersive prose."""

    async def generate_contribution(
        self,
        session: StorySession,
        context: Dict[str, Any],
    ) -> StoryBeat:
        """Generate contribution based on current mode using LLM."""
        story_so_far = context.get("story_so_far", "")

        if self.current_mode == "character" and self.active_character:
            user_prompt = f"""Story so far:
{story_so_far}

As {self.active_character.name}, write your next action or dialogue:"""
            char_pov = self.active_character.id
            beat_type = "dialogue"
        else:
            user_prompt = f"""Story so far:
{story_so_far}

Continue the narrative with the next scene:"""
            char_pov = None
            beat_type = "description"

        system_prompt = self.get_system_prompt(session)
        content = await self._call_llm(system_prompt, user_prompt, max_tokens=500)

        beat = StoryBeat(
            id=self._generate_beat_id(),
            content=content.strip(),
            beat_type=beat_type,
            author_agent=self.agent_id,
            character_pov=char_pov,
            phase=session.state.current_phase,
            tension_level=session.state.current_tension,
        )

        self.contributions += 1
        self.last_contribution = datetime.now(timezone.utc)

        return beat


# ============================================================================
# AGENT FACTORY
# ============================================================================

class StoryAgentFactory:
    """Factory for creating storytelling agents"""

    @staticmethod
    def create_narrator(
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ) -> NarratorAgent:
        """Create a narrator agent"""
        return NarratorAgent(personality=personality, model=model)

    @staticmethod
    def create_character_agent(
        character: StoryCharacter,
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ) -> CharacterAgent:
        """Create a character agent"""
        return CharacterAgent(character, personality, model=model)

    @staticmethod
    def create_director(
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ) -> DirectorAgent:
        """Create a director agent"""
        return DirectorAgent(personality=personality, model=model)

    @staticmethod
    def create_critic(
        personality: Optional[AgentPersonality] = None,
        model: str = "anthropic/claude-sonnet-4",
    ) -> CriticAgent:
        """Create a critic agent"""
        return CriticAgent(personality=personality, model=model)

    @staticmethod
    def create_ensemble(
        characters: List[StoryCharacter],
        include_narrator: bool = True,
        include_director: bool = True,
        include_critic: bool = False,
        model: str = "anthropic/claude-sonnet-4",
    ) -> Dict[str, StoryAgent]:
        """Create a complete ensemble of agents"""
        agents = {}

        if include_narrator:
            agents["narrator"] = StoryAgentFactory.create_narrator(model=model)

        if include_director:
            agents["director"] = StoryAgentFactory.create_director(model=model)

        if include_critic:
            agents["critic"] = StoryAgentFactory.create_critic(model=model)

        for char in characters:
            agent = StoryAgentFactory.create_character_agent(char, model=model)
            agents[agent.agent_id] = agent

        return agents
