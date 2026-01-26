"""
Collaborative Storytelling

Orchestrates multi-agent collaborative storytelling sessions
with real-time streaming and coordination.
"""

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

from .engine import (
    StorytellingEngine, StorySession, StoryConfig, StoryBeat,
    StoryCharacter, StoryState, StoryMode, StoryGenre,
    NarrativePhase, TensionLevel,
)
from .agents import (
    StoryAgent, NarratorAgent, CharacterAgent,
    DirectorAgent, CriticAgent, StoryAgentFactory,
    AgentPersonality,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONTRIBUTION TYPES
# ============================================================================

class ContributionType(str, Enum):
    """Types of story contributions"""
    NARRATIVE = "narrative"
    DIALOGUE = "dialogue"
    ACTION = "action"
    THOUGHT = "thought"
    DESCRIPTION = "description"
    DIRECTION = "direction"
    CRITIQUE = "critique"
    TRANSITION = "transition"


@dataclass
class StoryContribution:
    """A contribution to the collaborative story"""
    id: str
    content: str
    contribution_type: ContributionType

    # Attribution
    agent_id: str
    agent_name: str
    character_id: Optional[str] = None

    # Context
    round_number: int = 0
    phase: NarrativePhase = NarrativePhase.SETUP
    tension: TensionLevel = TensionLevel.CALM

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    word_count: int = 0

    # Quality
    score: float = 0.0
    feedback: List[str] = field(default_factory=list)

    # Voting (if enabled)
    votes: int = 0
    approved: bool = True

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class StoryRound:
    """A round of contributions"""
    round_number: int
    contributions: List[StoryContribution] = field(default_factory=list)

    # Round state
    phase: NarrativePhase = NarrativePhase.SETUP
    tension: TensionLevel = TensionLevel.CALM

    # Direction for this round
    director_guidance: Optional[str] = None
    focus_character: Optional[str] = None
    beat_type_requested: Optional[str] = None

    # Outcome
    selected_contribution: Optional[str] = None  # ID of chosen contribution
    combined_output: str = ""

    # Timing
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None

    def get_duration(self) -> float:
        """Get round duration in seconds"""
        if self.ended_at:
            return (self.ended_at - self.started_at).total_seconds()
        return 0.0


# ============================================================================
# COLLABORATIVE STORY
# ============================================================================

@dataclass
class CollaborativeStoryConfig:
    """Configuration for collaborative storytelling"""
    story_config: StoryConfig

    # Turn taking
    turn_mode: str = "round_robin"  # round_robin, director_assigned, voting, free_for_all
    agents_per_round: int = 3
    max_rounds: int = 30

    # Timing
    contribution_timeout: float = 30.0  # seconds
    round_timeout: float = 120.0

    # Quality control
    require_approval: bool = False
    min_contribution_score: float = 0.5
    enable_critique: bool = True

    # Interaction
    allow_agent_dialogue: bool = True
    allow_character_interaction: bool = True

    # Output
    stream_contributions: bool = True
    save_all_contributions: bool = True


class CollaborativeStory:
    """
    Orchestrates multi-agent collaborative storytelling.

    Manages agent turns, contribution collection, and story assembly.
    """

    def __init__(
        self,
        config: CollaborativeStoryConfig,
        engine: Optional[StorytellingEngine] = None,
    ):
        self.config = config
        self.engine = engine or StorytellingEngine()

        # Session
        self.session: Optional[StorySession] = None
        self.rounds: List[StoryRound] = []
        self.current_round: int = 0

        # Agents
        self.agents: Dict[str, StoryAgent] = {}
        self.turn_order: List[str] = []
        self.current_turn: int = 0

        # Callbacks
        self.on_contribution: Optional[Callable[[StoryContribution], None]] = None
        self.on_round_complete: Optional[Callable[[StoryRound], None]] = None
        self.on_story_complete: Optional[Callable[[StoryState], None]] = None

        # State
        self.is_running: bool = False
        self.is_paused: bool = False

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(
        self,
        characters: Optional[List[StoryCharacter]] = None,
        custom_agents: Optional[Dict[str, StoryAgent]] = None,
    ) -> StorySession:
        """Initialize the collaborative story session"""
        # Create session
        self.session = self.engine.create_session(self.config.story_config)
        self.session.status = "active"

        # Create or use agents
        if custom_agents:
            self.agents = custom_agents
        else:
            # Create default ensemble
            chars = characters or self._create_default_characters()
            self.agents = StoryAgentFactory.create_ensemble(
                characters=chars,
                include_narrator=self.config.story_config.include_narrator,
                include_director=self.config.story_config.include_director,
                include_critic=self.config.enable_critique,
            )

            # Add characters to session
            for char in chars:
                self.engine.add_character(self.session.id, char)

        # Set turn order
        self._setup_turn_order()

        logger.info(f"Initialized collaborative story: {self.session.id}")
        return self.session

    def _create_default_characters(self) -> List[StoryCharacter]:
        """Create default character set"""
        num_chars = self.config.story_config.num_character_agents
        genre = self.config.story_config.genre

        # Genre-appropriate character templates
        templates = {
            StoryGenre.DRAMA: [
                ("protagonist", "growth", ["purpose", "connection"]),
                ("antagonist", "fall", ["power", "control"]),
                ("mentor", "flat", ["wisdom", "guidance"]),
            ],
            StoryGenre.THRILLER: [
                ("protagonist", "testing", ["survival", "truth"]),
                ("antagonist", "corruption", ["power", "secrecy"]),
                ("ally", "growth", ["loyalty", "justice"]),
            ],
            StoryGenre.ROMANCE: [
                ("lead", "growth", ["love", "acceptance"]),
                ("love_interest", "growth", ["trust", "vulnerability"]),
                ("friend", "flat", ["support", "truth"]),
            ],
            StoryGenre.MYSTERY: [
                ("detective", "flat", ["truth", "justice"]),
                ("suspect", "redemption", ["redemption", "hiding"]),
                ("victim", "growth", ["survival", "revelation"]),
            ],
        }

        base_templates = templates.get(genre, templates[StoryGenre.DRAMA])

        characters = []
        for i in range(min(num_chars, len(base_templates))):
            role, arc, motivations = base_templates[i]
            char = StoryCharacter(
                id=f"char_{i}",
                name=f"Character {i+1}",  # Would be named by LLM in real implementation
                role=role,
                arc_type=arc,
                motivations=motivations,
            )
            characters.append(char)

        return characters

    def _setup_turn_order(self):
        """Set up the turn order for agents"""
        mode = self.config.story_config.mode

        if mode == StoryMode.ROUND_ROBIN:
            # Fixed rotation
            self.turn_order = list(self.agents.keys())

        elif mode == StoryMode.DIRECTOR_LED:
            # Director always goes first, then others
            self.turn_order = ["director"] if "director" in self.agents else []
            self.turn_order.extend(
                k for k in self.agents if k != "director"
            )

        else:
            # Default to all agents
            self.turn_order = list(self.agents.keys())

        self.session.turn_order = self.turn_order

    # ========================================================================
    # STORY EXECUTION
    # ========================================================================

    async def run(self) -> StoryState:
        """Run the collaborative storytelling session"""
        if not self.session:
            raise RuntimeError("Session not initialized. Call initialize() first.")

        self.is_running = True
        logger.info(f"Starting collaborative story: {self.session.id}")

        try:
            while self.is_running and self.current_round < self.config.max_rounds:
                if self.is_paused:
                    await asyncio.sleep(0.1)
                    continue

                # Run a round
                round_result = await self._run_round()
                self.rounds.append(round_result)

                # Check for story completion
                if self._should_end_story():
                    break

                self.current_round += 1

        except Exception as e:
            logger.error(f"Error in collaborative story: {e}")
            raise

        finally:
            self.is_running = False
            self.engine.complete_session(self.session.id)

            if self.on_story_complete:
                self.on_story_complete(self.session.state)

        return self.session.state

    async def _run_round(self) -> StoryRound:
        """Run a single round of contributions"""
        round_obj = StoryRound(
            round_number=self.current_round,
            phase=self.session.state.current_phase,
            tension=self.session.state.current_tension,
        )

        # Get director guidance if available
        if "director" in self.agents:
            guidance = await self._get_director_guidance()
            round_obj.director_guidance = guidance

        # Generate context
        context = self.engine.generate_prompt_context(self.session)
        context["round"] = self.current_round
        context["guidance"] = round_obj.director_guidance

        # Collect contributions
        agents_this_round = self._get_agents_for_round()

        contributions = await self._collect_contributions(agents_this_round, context)
        round_obj.contributions = contributions

        # Select and apply contribution
        selected = self._select_contribution(contributions)
        if selected:
            round_obj.selected_contribution = selected.id
            round_obj.combined_output = selected.content

            # Create and add beat
            beat = StoryBeat(
                id=selected.id,
                content=selected.content,
                beat_type=selected.contribution_type.value,
                author_agent=selected.agent_id,
                character_pov=selected.character_id,
                phase=self.session.state.current_phase,
                tension_level=self.session.state.current_tension,
            )
            self.engine.add_beat(self.session.id, beat)

        round_obj.ended_at = datetime.now(timezone.utc)

        if self.on_round_complete:
            self.on_round_complete(round_obj)

        return round_obj

    async def _get_director_guidance(self) -> str:
        """Get guidance from the director agent"""
        director = self.agents.get("director")
        if not director:
            return ""

        context = self.engine.generate_prompt_context(self.session)
        beat = await director.generate_contribution(self.session, context)
        return beat.content

    def _get_agents_for_round(self) -> List[str]:
        """Determine which agents contribute this round"""
        mode = self.config.story_config.mode

        if mode == StoryMode.ROUND_ROBIN:
            # Rotate through agents
            num_agents = self.config.agents_per_round
            start_idx = self.current_round % len(self.turn_order)
            agents = []
            for i in range(num_agents):
                idx = (start_idx + i) % len(self.turn_order)
                agents.append(self.turn_order[idx])
            return agents

        elif mode == StoryMode.DIRECTOR_LED:
            # Director picks based on guidance
            # (Simplified: just return narrative agents)
            return [k for k in self.turn_order if k not in ["director", "critic"]]

        else:
            # All agents
            return self.turn_order

    async def _collect_contributions(
        self,
        agent_ids: List[str],
        context: Dict[str, Any],
    ) -> List[StoryContribution]:
        """Collect contributions from specified agents"""
        contributions = []

        # Generate contributions (could be parallel with asyncio.gather)
        tasks = []
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent:
                tasks.append(self._get_contribution(agent, context))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, StoryContribution):
                contributions.append(result)

                if self.on_contribution:
                    self.on_contribution(result)

        return contributions

    async def _get_contribution(
        self,
        agent: StoryAgent,
        context: Dict[str, Any],
    ) -> StoryContribution:
        """Get a contribution from a single agent"""
        try:
            beat = await asyncio.wait_for(
                agent.generate_contribution(self.session, context),
                timeout=self.config.contribution_timeout,
            )

            # Determine contribution type
            type_map = {
                "description": ContributionType.DESCRIPTION,
                "dialogue": ContributionType.DIALOGUE,
                "action": ContributionType.ACTION,
                "thought": ContributionType.THOUGHT,
                "direction": ContributionType.DIRECTION,
                "critique": ContributionType.CRITIQUE,
                "transition": ContributionType.TRANSITION,
            }

            contribution = StoryContribution(
                id=beat.id,
                content=beat.content,
                contribution_type=type_map.get(beat.beat_type, ContributionType.NARRATIVE),
                agent_id=agent.agent_id,
                agent_name=agent.name,
                character_id=beat.character_pov,
                round_number=self.current_round,
                phase=self.session.state.current_phase,
                tension=self.session.state.current_tension,
            )

            # Score if critic available
            if self.config.enable_critique and "critic" in self.agents:
                critic = self.agents["critic"]
                if hasattr(critic, 'score_beat'):
                    scores = critic.score_beat(beat, self.session)
                    contribution.score = scores.get("overall", 0.5)

            return contribution

        except asyncio.TimeoutError:
            logger.warning(f"Contribution timeout for agent {agent.agent_id}")
            return StoryContribution(
                id=f"timeout_{agent.agent_id}_{self.current_round}",
                content="[Contribution timed out]",
                contribution_type=ContributionType.NARRATIVE,
                agent_id=agent.agent_id,
                agent_name=agent.name,
                round_number=self.current_round,
                score=0.0,
            )

    def _select_contribution(
        self,
        contributions: List[StoryContribution],
    ) -> Optional[StoryContribution]:
        """Select the best contribution for the story"""
        if not contributions:
            return None

        valid = [c for c in contributions if c.score >= self.config.min_contribution_score]

        if not valid:
            valid = contributions  # Fallback to all if none meet threshold

        # Selection strategies
        mode = self.config.turn_mode

        if mode == "voting":
            # Select by votes (placeholder - would need voting mechanism)
            return max(valid, key=lambda c: c.votes, default=valid[0])

        elif mode == "score":
            # Select by score
            return max(valid, key=lambda c: c.score, default=valid[0])

        else:
            # Round robin - take first from current turn
            return valid[0] if valid else None

    def _should_end_story(self) -> bool:
        """Check if the story should end"""
        state = self.session.state

        # Check beat count
        if len(state.beats) >= state.config.target_length:
            return True

        # Check phase
        if state.current_phase == NarrativePhase.DENOUEMENT:
            return True

        # Check max rounds
        if self.current_round >= self.config.max_rounds - 1:
            return True

        return False

    # ========================================================================
    # CONTROL
    # ========================================================================

    def pause(self):
        """Pause the story"""
        self.is_paused = True
        logger.info(f"Paused story: {self.session.id}")

    def resume(self):
        """Resume the story"""
        self.is_paused = False
        logger.info(f"Resumed story: {self.session.id}")

    def stop(self):
        """Stop the story"""
        self.is_running = False
        logger.info(f"Stopped story: {self.session.id}")

    # ========================================================================
    # STREAMING
    # ========================================================================

    async def stream_story(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream story events"""
        event_queue: asyncio.Queue = asyncio.Queue()

        # Set up callbacks
        original_contribution_cb = self.on_contribution
        original_round_cb = self.on_round_complete

        async def on_contribution(c: StoryContribution):
            await event_queue.put({
                "type": "contribution",
                "data": {
                    "id": c.id,
                    "content": c.content,
                    "agent": c.agent_name,
                    "character": c.character_id,
                    "round": c.round_number,
                }
            })
            if original_contribution_cb:
                original_contribution_cb(c)

        async def on_round(r: StoryRound):
            await event_queue.put({
                "type": "round_complete",
                "data": {
                    "round": r.round_number,
                    "phase": r.phase.value,
                    "tension": r.tension.value,
                    "output": r.combined_output,
                }
            })
            if original_round_cb:
                original_round_cb(r)

        # Wrap callbacks for async queue
        self.on_contribution = lambda c: asyncio.create_task(on_contribution(c))
        self.on_round_complete = lambda r: asyncio.create_task(on_round(r))

        # Start story in background
        story_task = asyncio.create_task(self.run())

        # Yield events
        try:
            while not story_task.done() or not event_queue.empty():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.5)
                    yield event
                except asyncio.TimeoutError:
                    continue

            # Final event
            yield {
                "type": "complete",
                "data": self.engine.get_story_analysis(self.session),
            }

        finally:
            # Restore callbacks
            self.on_contribution = original_contribution_cb
            self.on_round_complete = original_round_cb

    # ========================================================================
    # OUTPUT
    # ========================================================================

    def get_story_text(self) -> str:
        """Get the complete story as text"""
        return self.engine.render_story(self.session, format="text")

    def get_story_json(self) -> str:
        """Get the complete story as JSON"""
        return self.engine.render_story(self.session, format="json")

    def get_contributions_log(self) -> List[Dict[str, Any]]:
        """Get log of all contributions"""
        log = []
        for round_obj in self.rounds:
            for contrib in round_obj.contributions:
                log.append({
                    "round": contrib.round_number,
                    "agent": contrib.agent_name,
                    "type": contrib.contribution_type.value,
                    "content": contrib.content,
                    "score": contrib.score,
                    "selected": contrib.id == round_obj.selected_contribution,
                })
        return log


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_collaborative_story(
    title: str,
    genre: str = "drama",
    num_characters: int = 3,
    max_rounds: int = 20,
) -> CollaborativeStory:
    """Create a quick collaborative story setup"""
    story_config = StoryConfig(
        title=title,
        genre=StoryGenre(genre),
        mode=StoryMode.COLLABORATIVE,
        num_character_agents=num_characters,
        target_length=max_rounds,
        include_narrator=True,
        include_director=True,
    )

    config = CollaborativeStoryConfig(
        story_config=story_config,
        max_rounds=max_rounds,
        enable_critique=True,
    )

    return CollaborativeStory(config)


async def run_quick_story(
    title: str,
    genre: str = "drama",
    characters: Optional[List[StoryCharacter]] = None,
) -> str:
    """Run a quick collaborative story and return the result"""
    story = create_collaborative_story(title, genre)
    story.initialize(characters)
    await story.run()
    return story.get_story_text()


# ============================================================================
# DEMO
# ============================================================================

async def demo_collaborative():
    """Demo collaborative storytelling"""
    print("=== Collaborative Storytelling Demo ===\n")

    # Create story
    story = create_collaborative_story(
        title="The Crossroads",
        genre="drama",
        num_characters=2,
        max_rounds=5,
    )

    # Custom characters
    characters = [
        StoryCharacter(
            id="alex",
            name="Alex",
            role="protagonist",
            arc_type="growth",
            motivations=["redemption", "forgiveness"],
            backstory="A person seeking to make amends.",
        ),
        StoryCharacter(
            id="morgan",
            name="Morgan",
            role="antagonist",
            arc_type="testing",
            motivations=["justice", "truth"],
            backstory="Someone who was wronged in the past.",
        ),
    ]

    story.initialize(characters)

    # Run with streaming
    print("Generating story...\n")
    print("-" * 50)

    async for event in story.stream_story():
        if event["type"] == "contribution":
            data = event["data"]
            print(f"[{data['agent']}] {data['content'][:100]}...")
        elif event["type"] == "round_complete":
            data = event["data"]
            print(f"\n--- Round {data['round']} ({data['phase']}) ---\n")
        elif event["type"] == "complete":
            print("\n" + "=" * 50)
            print("STORY COMPLETE")
            print("=" * 50)

    # Output final story
    print("\n" + story.get_story_text())

    return story


if __name__ == "__main__":
    asyncio.run(demo_collaborative())
