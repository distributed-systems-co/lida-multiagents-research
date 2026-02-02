"""
Storytelling Engine

Core engine for multi-agent collaborative storytelling.
Manages story sessions, narrative flow, and agent coordination.
"""

import asyncio
import json
import random
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STORY CONFIGURATION
# ============================================================================

class StoryMode(str, Enum):
    """Storytelling modes"""
    COLLABORATIVE = "collaborative"      # Multiple agents build story together
    ROUND_ROBIN = "round_robin"          # Agents take turns
    DIRECTOR_LED = "director_led"        # Director agent guides story
    DEBATE_STYLE = "debate_style"        # Characters argue/debate
    INTERVIEW = "interview"              # Q&A style narrative
    IMPROV = "improv"                    # Free-form improvisation
    STRUCTURED = "structured"            # Following story beats
    EMERGENT = "emergent"                # Story emerges from agent interactions


class StoryGenre(str, Enum):
    """Story genres with associated conventions"""
    DRAMA = "drama"
    COMEDY = "comedy"
    TRAGEDY = "tragedy"
    THRILLER = "thriller"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    ADVENTURE = "adventure"
    HORROR = "horror"
    SCIFI = "scifi"
    FANTASY = "fantasy"
    SATIRE = "satire"
    POLITICAL = "political"
    PHILOSOPHICAL = "philosophical"
    HISTORICAL = "historical"
    DOCUMENTARY = "documentary"


class NarrativePhase(str, Enum):
    """Phases of narrative structure"""
    SETUP = "setup"
    INCITING_INCIDENT = "inciting_incident"
    RISING_ACTION = "rising_action"
    MIDPOINT = "midpoint"
    COMPLICATIONS = "complications"
    CRISIS = "crisis"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    DENOUEMENT = "denouement"


class TensionLevel(str, Enum):
    """Dramatic tension levels"""
    CALM = "calm"
    BUILDING = "building"
    MODERATE = "moderate"
    HIGH = "high"
    PEAK = "peak"
    RELEASING = "releasing"


@dataclass
class StoryConfig:
    """Configuration for a story session"""
    title: str = "Untitled Story"
    genre: StoryGenre = StoryGenre.DRAMA
    mode: StoryMode = StoryMode.COLLABORATIVE

    # Narrative settings
    target_length: int = 20          # Target number of story beats
    max_rounds: int = 50             # Maximum rounds before conclusion
    min_contribution_length: int = 50
    max_contribution_length: int = 500

    # Pacing
    tension_curve: str = "classic"   # classic, slow_burn, thriller, wave
    allow_flashbacks: bool = True
    allow_multiple_povs: bool = True

    # Agent settings
    num_character_agents: int = 3
    include_narrator: bool = True
    include_director: bool = True
    include_critic: bool = False

    # Themes and constraints
    themes: List[str] = field(default_factory=list)
    setting: str = ""
    time_period: str = ""
    constraints: List[str] = field(default_factory=list)

    # Interactive options
    allow_audience_input: bool = False
    branching_enabled: bool = False
    voting_enabled: bool = False

    # Output
    stream_output: bool = True
    save_drafts: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "genre": self.genre.value,
            "mode": self.mode.value,
            "target_length": self.target_length,
            "max_rounds": self.max_rounds,
            "themes": self.themes,
            "setting": self.setting,
            "time_period": self.time_period,
        }


# ============================================================================
# STORY ELEMENTS
# ============================================================================

@dataclass
class StoryBeat:
    """A single beat in the story"""
    id: str
    content: str
    beat_type: str  # action, dialogue, description, thought, transition

    # Attribution
    author_agent: str
    character_pov: Optional[str] = None

    # Narrative position
    phase: NarrativePhase = NarrativePhase.SETUP
    tension_level: TensionLevel = TensionLevel.CALM

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    word_count: int = 0

    # Relationships
    follows: Optional[str] = None  # Previous beat ID
    introduces: List[str] = field(default_factory=list)  # New elements
    references: List[str] = field(default_factory=list)  # Referenced elements

    # Quality metrics
    coherence_score: float = 0.0
    engagement_score: float = 0.0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class StoryCharacter:
    """A character in the story"""
    id: str
    name: str
    role: str  # protagonist, antagonist, supporting, etc.

    # Traits
    personality: Dict[str, float] = field(default_factory=dict)
    motivations: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)
    fears: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)

    # Arc
    arc_type: str = "growth"  # growth, fall, flat, redemption, corruption
    arc_progress: float = 0.0

    # State
    current_emotion: str = "neutral"
    current_location: str = ""
    relationships: Dict[str, str] = field(default_factory=dict)

    # History
    backstory: str = ""
    key_moments: List[str] = field(default_factory=list)

    def describe(self) -> str:
        traits = ", ".join(f"{k}: {v:.1f}" for k, v in self.personality.items())
        return f"{self.name} ({self.role}): {traits}"


@dataclass
class StoryWorld:
    """The world/setting of the story"""
    name: str = "The World"
    description: str = ""

    # Physical
    locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    current_location: str = ""

    # Temporal
    time_period: str = "present"
    current_time: str = ""
    timeline: List[Tuple[str, str]] = field(default_factory=list)  # (time, event)

    # Rules
    rules: List[str] = field(default_factory=list)
    magic_system: Optional[Dict[str, Any]] = None
    technology_level: str = "modern"

    # Atmosphere
    tone: str = "neutral"
    mood: str = "neutral"

    # Elements
    factions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    lore: List[str] = field(default_factory=list)


@dataclass
class PlotThread:
    """A plot thread or subplot"""
    id: str
    name: str
    description: str

    # Status
    status: str = "active"  # setup, active, climaxing, resolved, abandoned
    priority: int = 1  # 1 = main plot, higher = subplot

    # Characters involved
    characters: List[str] = field(default_factory=list)

    # Progress
    beats: List[str] = field(default_factory=list)  # Beat IDs
    setup_complete: bool = False
    resolution: Optional[str] = None

    # Connections
    parent_thread: Optional[str] = None
    child_threads: List[str] = field(default_factory=list)


# ============================================================================
# STORY STATE
# ============================================================================

@dataclass
class StoryState:
    """Complete state of the story"""
    id: str
    config: StoryConfig

    # Narrative elements
    beats: List[StoryBeat] = field(default_factory=list)
    characters: Dict[str, StoryCharacter] = field(default_factory=dict)
    world: StoryWorld = field(default_factory=StoryWorld)
    plot_threads: Dict[str, PlotThread] = field(default_factory=dict)

    # Progress
    current_phase: NarrativePhase = NarrativePhase.SETUP
    current_tension: TensionLevel = TensionLevel.CALM
    current_round: int = 0

    # Metrics
    word_count: int = 0
    beat_count: int = 0

    # Tracking
    introduced_elements: Set[str] = field(default_factory=set)
    unresolved_hooks: List[str] = field(default_factory=list)
    chekhov_guns: List[Tuple[str, bool]] = field(default_factory=list)  # (element, fired)

    # Quality
    coherence_score: float = 0.0
    pacing_score: float = 0.0
    engagement_score: float = 0.0

    def get_summary(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.config.title,
            "genre": self.config.genre.value,
            "phase": self.current_phase.value,
            "tension": self.current_tension.value,
            "beats": len(self.beats),
            "words": self.word_count,
            "characters": len(self.characters),
            "rounds": self.current_round,
        }


# ============================================================================
# STORY SESSION
# ============================================================================

@dataclass
class StorySession:
    """An active storytelling session"""
    id: str
    state: StoryState

    # Agents
    agents: Dict[str, Any] = field(default_factory=dict)
    turn_order: List[str] = field(default_factory=list)
    current_turn: int = 0

    # Session state
    status: str = "initializing"  # initializing, active, paused, completed, abandoned
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None

    # History
    contributions: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)

    # Branching
    branch_points: List[Dict[str, Any]] = field(default_factory=list)
    current_branch: str = "main"

    def get_current_agent(self) -> Optional[str]:
        if not self.turn_order:
            return None
        return self.turn_order[self.current_turn % len(self.turn_order)]

    def advance_turn(self):
        self.current_turn += 1
        if self.current_turn >= len(self.turn_order):
            self.current_turn = 0
            self.state.current_round += 1


# ============================================================================
# STORYTELLING ENGINE
# ============================================================================

class StorytellingEngine:
    """
    Core engine for multi-agent collaborative storytelling.

    Manages story sessions, coordinates agents, and ensures
    narrative coherence across contributions.
    """

    # Tension curves for different pacing styles
    TENSION_CURVES = {
        "classic": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.7, 0.4, 0.2],
        "slow_burn": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9, 0.95, 0.5, 0.2],
        "thriller": [0.5, 0.6, 0.7, 0.8, 0.6, 0.8, 0.9, 0.7, 0.95, 1.0, 0.6, 0.3],
        "wave": [0.3, 0.6, 0.4, 0.7, 0.5, 0.8, 0.6, 0.9, 0.7, 0.95, 0.5, 0.2],
    }

    # Phase progression
    PHASE_ORDER = [
        NarrativePhase.SETUP,
        NarrativePhase.INCITING_INCIDENT,
        NarrativePhase.RISING_ACTION,
        NarrativePhase.MIDPOINT,
        NarrativePhase.COMPLICATIONS,
        NarrativePhase.CRISIS,
        NarrativePhase.CLIMAX,
        NarrativePhase.FALLING_ACTION,
        NarrativePhase.RESOLUTION,
        NarrativePhase.DENOUEMENT,
    ]

    def __init__(self):
        self.sessions: Dict[str, StorySession] = {}
        self.completed_stories: List[str] = []
        self._session_counter = 0

        # Genre conventions
        self.genre_conventions = self._load_genre_conventions()

        # Callbacks
        self.on_beat_added: Optional[Callable] = None
        self.on_phase_change: Optional[Callable] = None
        self.on_story_complete: Optional[Callable] = None

    def _generate_session_id(self) -> str:
        self._session_counter += 1
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"STORY-{timestamp}-{self._session_counter:04d}"

    def _load_genre_conventions(self) -> Dict[StoryGenre, Dict[str, Any]]:
        """Load genre-specific storytelling conventions"""
        return {
            StoryGenre.DRAMA: {
                "pacing": "classic",
                "focus": ["character", "emotion", "relationships"],
                "typical_beats": ["revelation", "confrontation", "reconciliation"],
                "tone": "serious",
                "tension_multiplier": 1.0,
            },
            StoryGenre.COMEDY: {
                "pacing": "wave",
                "focus": ["timing", "irony", "subversion"],
                "typical_beats": ["setup", "escalation", "payoff"],
                "tone": "light",
                "tension_multiplier": 0.7,
            },
            StoryGenre.THRILLER: {
                "pacing": "thriller",
                "focus": ["suspense", "twists", "pacing"],
                "typical_beats": ["threat", "chase", "revelation", "escape"],
                "tone": "tense",
                "tension_multiplier": 1.3,
            },
            StoryGenre.MYSTERY: {
                "pacing": "slow_burn",
                "focus": ["clues", "deduction", "misdirection"],
                "typical_beats": ["discovery", "investigation", "revelation"],
                "tone": "intriguing",
                "tension_multiplier": 1.1,
            },
            StoryGenre.ROMANCE: {
                "pacing": "wave",
                "focus": ["chemistry", "obstacles", "emotion"],
                "typical_beats": ["meeting", "tension", "separation", "reunion"],
                "tone": "romantic",
                "tension_multiplier": 0.9,
            },
            StoryGenre.SCIFI: {
                "pacing": "classic",
                "focus": ["ideas", "technology", "consequences"],
                "typical_beats": ["discovery", "exploration", "crisis", "resolution"],
                "tone": "speculative",
                "tension_multiplier": 1.0,
            },
            StoryGenre.FANTASY: {
                "pacing": "classic",
                "focus": ["worldbuilding", "quest", "magic"],
                "typical_beats": ["call", "journey", "trials", "transformation"],
                "tone": "epic",
                "tension_multiplier": 1.0,
            },
            StoryGenre.HORROR: {
                "pacing": "slow_burn",
                "focus": ["atmosphere", "dread", "threat"],
                "typical_beats": ["unease", "investigation", "encounter", "survival"],
                "tone": "dark",
                "tension_multiplier": 1.4,
            },
            StoryGenre.POLITICAL: {
                "pacing": "classic",
                "focus": ["power", "intrigue", "consequences"],
                "typical_beats": ["maneuvering", "betrayal", "confrontation"],
                "tone": "serious",
                "tension_multiplier": 1.1,
            },
            StoryGenre.PHILOSOPHICAL: {
                "pacing": "slow_burn",
                "focus": ["ideas", "dialogue", "meaning"],
                "typical_beats": ["question", "exploration", "insight"],
                "tone": "contemplative",
                "tension_multiplier": 0.8,
            },
        }

    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================

    def create_session(self, config: StoryConfig) -> StorySession:
        """Create a new storytelling session"""
        session_id = self._generate_session_id()

        state = StoryState(
            id=session_id,
            config=config,
            world=StoryWorld(
                name=config.setting or "The World",
                time_period=config.time_period or "present",
            ),
        )

        # Create main plot thread
        main_thread = PlotThread(
            id="main",
            name="Main Plot",
            description=f"The main story of {config.title}",
            priority=1,
        )
        state.plot_threads["main"] = main_thread

        session = StorySession(
            id=session_id,
            state=state,
        )

        self.sessions[session_id] = session
        logger.info(f"Created story session: {session_id}")

        return session

    def get_session(self, session_id: str) -> Optional[StorySession]:
        """Get a session by ID"""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions"""
        return [
            {
                "id": s.id,
                "title": s.state.config.title,
                "status": s.status,
                "beats": len(s.state.beats),
                "started": s.started_at.isoformat(),
            }
            for s in self.sessions.values()
        ]

    # ========================================================================
    # STORY BUILDING
    # ========================================================================

    def add_character(self, session_id: str, character: StoryCharacter) -> bool:
        """Add a character to the story"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.state.characters[character.id] = character
        session.state.introduced_elements.add(f"character:{character.id}")

        logger.info(f"Added character {character.name} to session {session_id}")
        return True

    def add_beat(self, session_id: str, beat: StoryBeat) -> bool:
        """Add a story beat"""
        session = self.get_session(session_id)
        if not session:
            return False

        # Link to previous beat
        if session.state.beats:
            beat.follows = session.state.beats[-1].id

        # Update state
        session.state.beats.append(beat)
        session.state.beat_count += 1
        session.state.word_count += beat.word_count

        # Track introduced elements
        for element in beat.introduces:
            session.state.introduced_elements.add(element)

        # Update tension and phase
        self._update_narrative_position(session, beat)

        # Callback
        if self.on_beat_added:
            self.on_beat_added(session, beat)

        logger.debug(f"Added beat {beat.id} to session {session_id}")
        return True

    def _update_narrative_position(self, session: StorySession, beat: StoryBeat):
        """Update narrative phase and tension based on progress"""
        progress = len(session.state.beats) / session.state.config.target_length

        # Determine phase
        phase_index = int(progress * len(self.PHASE_ORDER))
        phase_index = min(phase_index, len(self.PHASE_ORDER) - 1)
        new_phase = self.PHASE_ORDER[phase_index]

        if new_phase != session.state.current_phase:
            old_phase = session.state.current_phase
            session.state.current_phase = new_phase
            beat.phase = new_phase

            if self.on_phase_change:
                self.on_phase_change(session, old_phase, new_phase)

        # Determine tension
        curve_name = session.state.config.tension_curve
        curve = self.TENSION_CURVES.get(curve_name, self.TENSION_CURVES["classic"])

        curve_index = int(progress * len(curve))
        curve_index = min(curve_index, len(curve) - 1)
        target_tension = curve[curve_index]

        # Apply genre multiplier
        genre = session.state.config.genre
        if genre in self.genre_conventions:
            target_tension *= self.genre_conventions[genre]["tension_multiplier"]

        # Map to tension level
        if target_tension < 0.2:
            session.state.current_tension = TensionLevel.CALM
        elif target_tension < 0.4:
            session.state.current_tension = TensionLevel.BUILDING
        elif target_tension < 0.6:
            session.state.current_tension = TensionLevel.MODERATE
        elif target_tension < 0.8:
            session.state.current_tension = TensionLevel.HIGH
        else:
            session.state.current_tension = TensionLevel.PEAK

        beat.tension_level = session.state.current_tension

    # ========================================================================
    # STORY GENERATION
    # ========================================================================

    def generate_prompt_context(self, session: StorySession) -> Dict[str, Any]:
        """Generate context for agent prompts"""
        state = session.state
        config = state.config

        # Get recent beats
        recent_beats = state.beats[-5:] if state.beats else []
        story_so_far = "\n\n".join(b.content for b in recent_beats)

        # Get active characters
        active_chars = [
            {"name": c.name, "role": c.role, "emotion": c.current_emotion}
            for c in state.characters.values()
        ]

        # Get genre conventions
        conventions = self.genre_conventions.get(config.genre, {})

        context = {
            "story_title": config.title,
            "genre": config.genre.value,
            "mode": config.mode.value,
            "setting": state.world.name,
            "time_period": state.world.time_period,
            "themes": config.themes,

            "current_phase": state.current_phase.value,
            "tension_level": state.current_tension.value,
            "beat_count": state.beat_count,
            "target_beats": config.target_length,
            "progress": state.beat_count / config.target_length,

            "story_so_far": story_so_far,
            "characters": active_chars,
            "current_location": state.world.current_location,

            "conventions": conventions,
            "constraints": config.constraints,

            "unresolved_hooks": state.unresolved_hooks,
        }

        return context

    def get_pacing_guidance(self, session: StorySession) -> Dict[str, Any]:
        """Get pacing guidance for the current story position"""
        state = session.state
        progress = state.beat_count / state.config.target_length

        guidance = {
            "phase": state.current_phase.value,
            "tension": state.current_tension.value,
            "progress_percent": int(progress * 100),
        }

        # Phase-specific guidance
        if state.current_phase == NarrativePhase.SETUP:
            guidance["focus"] = "Establish characters, setting, and situation"
            guidance["avoid"] = "Major conflict or revelation"
            guidance["introduce"] = ["character traits", "world details", "potential conflicts"]

        elif state.current_phase == NarrativePhase.INCITING_INCIDENT:
            guidance["focus"] = "Disrupt the status quo, create stakes"
            guidance["avoid"] = "Premature resolution"
            guidance["introduce"] = ["central conflict", "story question", "urgency"]

        elif state.current_phase == NarrativePhase.RISING_ACTION:
            guidance["focus"] = "Escalate tension, develop characters"
            guidance["avoid"] = "Resolution or backing down"
            guidance["introduce"] = ["complications", "character depth", "relationships"]

        elif state.current_phase == NarrativePhase.MIDPOINT:
            guidance["focus"] = "Shift perspective, raise stakes"
            guidance["avoid"] = "Maintaining status quo"
            guidance["introduce"] = ["revelation", "commitment", "point of no return"]

        elif state.current_phase == NarrativePhase.COMPLICATIONS:
            guidance["focus"] = "Increase obstacles, test characters"
            guidance["avoid"] = "Easy solutions"
            guidance["introduce"] = ["setbacks", "betrayals", "difficult choices"]

        elif state.current_phase == NarrativePhase.CRISIS:
            guidance["focus"] = "Push to breaking point"
            guidance["avoid"] = "Premature relief"
            guidance["introduce"] = ["darkest moment", "sacrifice", "desperation"]

        elif state.current_phase == NarrativePhase.CLIMAX:
            guidance["focus"] = "Decisive confrontation"
            guidance["avoid"] = "New complications"
            guidance["introduce"] = ["resolution of conflict", "character transformation"]

        elif state.current_phase == NarrativePhase.FALLING_ACTION:
            guidance["focus"] = "Show aftermath, tie up threads"
            guidance["avoid"] = "New major conflicts"
            guidance["introduce"] = ["consequences", "new equilibrium"]

        elif state.current_phase == NarrativePhase.RESOLUTION:
            guidance["focus"] = "Resolve remaining questions"
            guidance["avoid"] = "Undermining climax"
            guidance["introduce"] = ["final character moments", "thematic closure"]

        elif state.current_phase == NarrativePhase.DENOUEMENT:
            guidance["focus"] = "Emotional closure"
            guidance["avoid"] = "New information"
            guidance["introduce"] = ["final image", "emotional resonance"]

        return guidance

    # ========================================================================
    # COHERENCE AND QUALITY
    # ========================================================================

    def evaluate_coherence(self, session: StorySession) -> Dict[str, Any]:
        """Evaluate story coherence"""
        state = session.state
        scores = {}

        # Character consistency
        if state.characters:
            char_mentions = defaultdict(int)
            for beat in state.beats:
                for char_id in state.characters:
                    if state.characters[char_id].name.lower() in beat.content.lower():
                        char_mentions[char_id] += 1

            # All characters should be mentioned
            mentioned = sum(1 for c in char_mentions.values() if c > 0)
            scores["character_presence"] = mentioned / len(state.characters)
        else:
            scores["character_presence"] = 1.0

        # Plot thread resolution
        if state.plot_threads:
            resolved = sum(1 for t in state.plot_threads.values() if t.status == "resolved")
            scores["plot_resolution"] = resolved / len(state.plot_threads)
        else:
            scores["plot_resolution"] = 1.0

        # Pacing
        actual_beats = len(state.beats)
        target_beats = state.config.target_length
        pacing_diff = abs(actual_beats - target_beats) / target_beats
        scores["pacing"] = max(0, 1 - pacing_diff)

        # Chekhov's guns
        if state.chekhov_guns:
            fired = sum(1 for _, f in state.chekhov_guns if f)
            scores["setups_resolved"] = fired / len(state.chekhov_guns)
        else:
            scores["setups_resolved"] = 1.0

        # Overall
        scores["overall"] = sum(scores.values()) / len(scores)

        return scores

    def get_story_analysis(self, session: StorySession) -> Dict[str, Any]:
        """Get comprehensive story analysis"""
        state = session.state

        # Word statistics
        word_counts = [b.word_count for b in state.beats]
        avg_beat_length = sum(word_counts) / len(word_counts) if word_counts else 0

        # Phase distribution
        phase_counts = defaultdict(int)
        for beat in state.beats:
            phase_counts[beat.phase.value] += 1

        # Tension distribution
        tension_counts = defaultdict(int)
        for beat in state.beats:
            tension_counts[beat.tension_level.value] += 1

        # Agent contributions
        agent_beats = defaultdict(int)
        for beat in state.beats:
            agent_beats[beat.author_agent] += 1

        return {
            "summary": state.get_summary(),
            "coherence": self.evaluate_coherence(session),
            "statistics": {
                "total_words": state.word_count,
                "total_beats": len(state.beats),
                "avg_beat_length": avg_beat_length,
                "characters": len(state.characters),
                "plot_threads": len(state.plot_threads),
            },
            "distribution": {
                "phases": dict(phase_counts),
                "tension": dict(tension_counts),
                "agents": dict(agent_beats),
            },
        }

    # ========================================================================
    # OUTPUT GENERATION
    # ========================================================================

    def render_story(self, session: StorySession, format: str = "text") -> str:
        """Render the complete story"""
        state = session.state

        if format == "text":
            lines = [
                f"# {state.config.title}",
                f"*A {state.config.genre.value} story*",
                "",
                "---",
                "",
            ]

            for beat in state.beats:
                lines.append(beat.content)
                lines.append("")

            lines.extend([
                "---",
                "",
                f"*{state.word_count} words | {len(state.beats)} beats*",
            ])

            return "\n".join(lines)

        elif format == "json":
            return json.dumps({
                "title": state.config.title,
                "genre": state.config.genre.value,
                "beats": [
                    {
                        "content": b.content,
                        "author": b.author_agent,
                        "phase": b.phase.value,
                        "tension": b.tension_level.value,
                    }
                    for b in state.beats
                ],
                "characters": {
                    cid: {"name": c.name, "role": c.role}
                    for cid, c in state.characters.items()
                },
                "statistics": {
                    "words": state.word_count,
                    "beats": len(state.beats),
                },
            }, indent=2)

        elif format == "screenplay":
            lines = [
                f"TITLE: {state.config.title}",
                f"GENRE: {state.config.genre.value.upper()}",
                "",
                "FADE IN:",
                "",
            ]

            for beat in state.beats:
                if beat.beat_type == "dialogue":
                    lines.append(beat.content.upper())
                elif beat.beat_type == "action":
                    lines.append(beat.content)
                else:
                    lines.append(f"({beat.content})")
                lines.append("")

            lines.append("FADE OUT.")
            return "\n".join(lines)

        return ""

    def complete_session(self, session_id: str) -> bool:
        """Mark a session as completed"""
        session = self.get_session(session_id)
        if not session:
            return False

        session.status = "completed"
        session.ended_at = datetime.now(timezone.utc)
        self.completed_stories.append(session_id)

        if self.on_story_complete:
            self.on_story_complete(session)

        logger.info(f"Completed story session: {session_id}")
        return True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_story_engine() -> StorytellingEngine:
    """Create a new storytelling engine"""
    return StorytellingEngine()


def quick_story_config(
    title: str,
    genre: str = "drama",
    mode: str = "collaborative",
    num_agents: int = 3,
) -> StoryConfig:
    """Create a quick story configuration"""
    return StoryConfig(
        title=title,
        genre=StoryGenre(genre),
        mode=StoryMode(mode),
        num_character_agents=num_agents,
    )
