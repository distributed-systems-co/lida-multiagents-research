"""
Narrative Intelligence Engine

Advanced narrative simulation including:
- Story structure and plot generation
- Character arc modeling with transformation
- Dramatic tension and pacing
- Theme extraction and development
- Narrative coherence evaluation
- Multi-protagonist storylines
- Branching narratives and counterfactuals
- Genre-aware story generation

Based on:
- Narratology (Propp, Campbell, McKee)
- Computational Creativity
- Story Understanding AI
- Interactive Drama Research
"""

import asyncio
import random
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


# ============================================================================
# STORY STRUCTURE
# ============================================================================

class StoryPhase(Enum):
    """Phases of story structure (based on multiple theories)"""
    # Three-Act Structure
    SETUP = "setup"
    CONFRONTATION = "confrontation"
    CLIMAX = "climax"
    RESOLUTION = "resolution"

    # Hero's Journey additions
    ORDINARY_WORLD = "ordinary_world"
    CALL_TO_ADVENTURE = "call_to_adventure"
    REFUSAL_OF_CALL = "refusal"
    MEETING_MENTOR = "mentor"
    CROSSING_THRESHOLD = "threshold"
    TESTS_ALLIES_ENEMIES = "tests"
    APPROACH_INNERMOST_CAVE = "approach"
    ORDEAL = "ordeal"
    REWARD = "reward"
    ROAD_BACK = "road_back"
    RESURRECTION = "resurrection"
    RETURN_WITH_ELIXIR = "return"


class PlotPointType(Enum):
    """Types of plot points"""
    INCITING_INCIDENT = "inciting_incident"
    PLOT_POINT_1 = "plot_point_1"  # End of Act 1
    MIDPOINT = "midpoint"
    PLOT_POINT_2 = "plot_point_2"  # End of Act 2
    CLIMAX = "climax"
    DENOUEMENT = "denouement"

    # Additional structural points
    PINCH_POINT_1 = "pinch_1"
    PINCH_POINT_2 = "pinch_2"
    DARK_MOMENT = "dark_moment"
    REVELATION = "revelation"
    REVERSAL = "reversal"


class ConflictType(Enum):
    """Types of narrative conflict"""
    PERSON_VS_PERSON = "person_vs_person"
    PERSON_VS_SELF = "person_vs_self"
    PERSON_VS_NATURE = "person_vs_nature"
    PERSON_VS_SOCIETY = "person_vs_society"
    PERSON_VS_TECHNOLOGY = "person_vs_technology"
    PERSON_VS_SUPERNATURAL = "person_vs_supernatural"
    PERSON_VS_FATE = "person_vs_fate"


class Genre(Enum):
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


# ============================================================================
# CHARACTER ARCS
# ============================================================================

class ArcType(Enum):
    """Types of character arcs"""
    POSITIVE_CHANGE = "positive"  # Character grows/improves
    NEGATIVE_CHANGE = "negative"  # Character falls/corrupts
    FLAT = "flat"  # Character stays same, changes world
    DISILLUSIONMENT = "disillusionment"  # Positive to negative worldview
    MATURATION = "maturation"  # Coming of age
    REDEMPTION = "redemption"  # Bad to good
    CORRUPTION = "corruption"  # Good to bad
    TESTING = "testing"  # Beliefs tested and reaffirmed


@dataclass
class CharacterWant:
    """What a character consciously wants"""
    description: str
    intensity: float = 0.7  # How badly they want it
    achievable: bool = True
    moral: bool = True  # Is the want morally good?


@dataclass
class CharacterNeed:
    """What a character truly needs (often unconscious)"""
    description: str
    awareness: float = 0.2  # How aware they are of this need
    satisfied: bool = False


@dataclass
class CharacterFlaw:
    """Character's moral/psychological flaw"""
    description: str
    severity: float = 0.5
    origin: str = ""  # Ghost/wound that caused the flaw
    manifestations: List[str] = field(default_factory=list)


@dataclass
class CharacterGhost:
    """Past trauma/wound that drives character"""
    description: str
    when: str = ""
    impact: float = 0.7
    resolved: bool = False


@dataclass
class NarrativeCharacter:
    """A character in the narrative with full psychological profile"""
    id: str
    name: str
    role: str  # protagonist, antagonist, mentor, etc.

    # Core psychology
    want: CharacterWant = field(default_factory=lambda: CharacterWant("undefined"))
    need: CharacterNeed = field(default_factory=lambda: CharacterNeed("undefined"))
    flaw: CharacterFlaw = field(default_factory=lambda: CharacterFlaw("undefined"))
    ghost: Optional[CharacterGhost] = None

    # Arc
    arc_type: ArcType = ArcType.POSITIVE_CHANGE
    arc_progress: float = 0.0  # 0-1, how far through arc
    transformation_moments: List[str] = field(default_factory=list)

    # Beliefs that can change
    beliefs: Dict[str, float] = field(default_factory=dict)  # Topic -> conviction

    # Relationships
    relationships: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # State
    emotional_state: Dict[str, float] = field(default_factory=dict)
    knowledge: Set[str] = field(default_factory=set)
    current_goal: str = ""

    def get_internal_conflict(self) -> float:
        """Calculate internal conflict between want and need"""
        if not self.want or not self.need:
            return 0.0
        # Higher when want and need are misaligned and flaw is severe
        return self.flaw.severity * (1 - self.need.awareness) * self.want.intensity


@dataclass
class CharacterRelationship:
    """Relationship between characters"""
    character1: str
    character2: str
    relationship_type: str  # ally, enemy, lover, mentor, etc.

    trust: float = 0.5
    affection: float = 0.5
    respect: float = 0.5
    power_balance: float = 0.5  # 0=char1 dominant, 1=char2 dominant

    history: List[str] = field(default_factory=list)
    secrets: List[str] = field(default_factory=list)  # What one knows that other doesn't

    tension: float = 0.0
    chemistry: float = 0.5


# ============================================================================
# DRAMATIC TENSION
# ============================================================================

@dataclass
class DramaticQuestion:
    """Central question driving narrative tension"""
    question: str
    stakes: str  # What's at risk if question answered negatively
    urgency: float = 0.5
    answered: bool = False
    answer: Optional[bool] = None


@dataclass
class TensionElement:
    """An element contributing to dramatic tension"""
    id: str
    element_type: str  # ticking_clock, secret, threat, mystery, etc.
    description: str
    intensity: float = 0.5
    resolution_condition: str = ""
    resolved: bool = False


class TensionCurve:
    """Manages dramatic tension throughout the story"""

    def __init__(self):
        self.current_tension: float = 0.2
        self.target_tension: float = 0.2
        self.elements: List[TensionElement] = []
        self.history: List[Tuple[int, float]] = []
        self.phase: StoryPhase = StoryPhase.SETUP

        # Ideal tension curves for different phases
        self.phase_targets = {
            StoryPhase.SETUP: 0.2,
            StoryPhase.ORDINARY_WORLD: 0.15,
            StoryPhase.CALL_TO_ADVENTURE: 0.35,
            StoryPhase.CROSSING_THRESHOLD: 0.4,
            StoryPhase.TESTS_ALLIES_ENEMIES: 0.5,
            StoryPhase.APPROACH_INNERMOST_CAVE: 0.65,
            StoryPhase.ORDEAL: 0.85,
            StoryPhase.CONFRONTATION: 0.7,
            StoryPhase.CLIMAX: 0.95,
            StoryPhase.RESOLUTION: 0.3,
            StoryPhase.RETURN_WITH_ELIXIR: 0.2
        }

    def add_element(self, element: TensionElement):
        """Add a tension-generating element"""
        self.elements.append(element)
        self._recalculate()

    def resolve_element(self, element_id: str):
        """Resolve a tension element"""
        for elem in self.elements:
            if elem.id == element_id:
                elem.resolved = True
                break
        self._recalculate()

    def set_phase(self, phase: StoryPhase):
        """Set the current story phase"""
        self.phase = phase
        self.target_tension = self.phase_targets.get(phase, 0.5)

    def _recalculate(self):
        """Recalculate current tension from elements"""
        active_elements = [e for e in self.elements if not e.resolved]
        if not active_elements:
            self.current_tension = self.target_tension * 0.5
        else:
            element_tension = sum(e.intensity for e in active_elements)
            self.current_tension = min(1.0, element_tension / len(active_elements))

    def update(self, beat_number: int):
        """Update tension with story beat"""
        # Smooth transition towards target
        diff = self.target_tension - self.current_tension
        self.current_tension += diff * 0.1
        self.history.append((beat_number, self.current_tension))

    def get_pacing_suggestion(self) -> str:
        """Suggest pacing adjustment based on tension"""
        diff = self.target_tension - self.current_tension
        if diff > 0.2:
            return "increase_tension"
        elif diff < -0.2:
            return "release_tension"
        else:
            return "maintain"


# ============================================================================
# PLOT EVENTS
# ============================================================================

@dataclass
class Beat:
    """A single story beat (smallest unit of story)"""
    id: str
    description: str
    beat_type: str  # action, reaction, revelation, decision, etc.

    characters_involved: List[str] = field(default_factory=list)
    location: str = ""

    tension_change: float = 0.0  # -1 to 1
    arc_progress: Dict[str, float] = field(default_factory=dict)  # Character -> progress

    causes: List[str] = field(default_factory=list)  # Beat IDs that led to this
    consequences: List[str] = field(default_factory=list)  # Beat IDs this leads to

    emotional_beats: Dict[str, str] = field(default_factory=dict)  # Character -> emotion

    timestamp: int = 0


@dataclass
class Scene:
    """A scene containing multiple beats"""
    id: str
    goal: str  # What the scene needs to accomplish
    location: str
    characters: List[str]

    beats: List[Beat] = field(default_factory=list)

    # Scene-level dynamics
    opening_value: float = 0.0  # Value at stake (0-1)
    closing_value: float = 0.0  # How it ends

    conflict_type: Optional[ConflictType] = None
    polarity_shift: bool = False  # Did value flip during scene?

    turning_point: Optional[str] = None  # Key moment


@dataclass
class Sequence:
    """A sequence of scenes building toward a goal"""
    id: str
    goal: str
    scenes: List[Scene] = field(default_factory=list)
    plot_point: Optional[PlotPointType] = None


@dataclass
class Act:
    """An act containing sequences"""
    id: str
    act_number: int
    goal: str
    sequences: List[Sequence] = field(default_factory=list)

    opening_state: Dict[str, Any] = field(default_factory=dict)
    closing_state: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# THEMES
# ============================================================================

@dataclass
class Theme:
    """A thematic element of the story"""
    id: str
    statement: str  # Thematic statement (e.g., "Love conquers fear")
    question: str  # Thematic question (e.g., "Can love conquer fear?")

    motifs: List[str] = field(default_factory=list)  # Recurring symbols

    # How characters embody theme
    character_perspectives: Dict[str, str] = field(default_factory=dict)

    # Development through story
    development_beats: List[str] = field(default_factory=list)

    resolution: Optional[str] = None


# ============================================================================
# STORY STATE
# ============================================================================

@dataclass
class StoryState:
    """Complete state of the narrative"""
    # Structure
    genre: Genre = Genre.DRAMA
    current_phase: StoryPhase = StoryPhase.SETUP
    current_act: int = 1
    beat_count: int = 0

    # Characters
    characters: Dict[str, NarrativeCharacter] = field(default_factory=dict)
    relationships: Dict[str, CharacterRelationship] = field(default_factory=dict)

    # Plot
    acts: List[Act] = field(default_factory=list)
    current_scene: Optional[Scene] = None

    # Dramatic elements
    dramatic_questions: List[DramaticQuestion] = field(default_factory=list)
    tension_curve: TensionCurve = field(default_factory=TensionCurve)

    # Themes
    themes: List[Theme] = field(default_factory=list)

    # World state
    world_facts: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Beat] = field(default_factory=list)

    # Branching
    branch_points: List[Dict[str, Any]] = field(default_factory=list)
    counterfactuals: Dict[str, "StoryState"] = field(default_factory=dict)


# ============================================================================
# NARRATIVE ENGINE
# ============================================================================

class NarrativeEngine:
    """
    Engine for generating and managing narratives.
    """

    def __init__(self, genre: Genre = Genre.DRAMA):
        self.state = StoryState(genre=genre)
        self.genre_conventions = self._load_genre_conventions(genre)
        self.beat_generators: Dict[str, Callable] = {}
        self.coherence_threshold = 0.7

        # Register default beat generators
        self._register_default_generators()

    def _load_genre_conventions(self, genre: Genre) -> Dict[str, Any]:
        """Load genre-specific conventions"""
        conventions = {
            Genre.DRAMA: {
                "pacing": "moderate",
                "emotional_range": "wide",
                "typical_conflicts": [ConflictType.PERSON_VS_SELF, ConflictType.PERSON_VS_PERSON],
                "typical_arcs": [ArcType.POSITIVE_CHANGE, ArcType.NEGATIVE_CHANGE],
                "tone": "serious"
            },
            Genre.COMEDY: {
                "pacing": "fast",
                "emotional_range": "light_to_moderate",
                "typical_conflicts": [ConflictType.PERSON_VS_SOCIETY, ConflictType.PERSON_VS_PERSON],
                "typical_arcs": [ArcType.POSITIVE_CHANGE, ArcType.FLAT],
                "tone": "humorous"
            },
            Genre.TRAGEDY: {
                "pacing": "slow_build",
                "emotional_range": "heavy",
                "typical_conflicts": [ConflictType.PERSON_VS_FATE, ConflictType.PERSON_VS_SELF],
                "typical_arcs": [ArcType.NEGATIVE_CHANGE, ArcType.CORRUPTION],
                "tone": "somber"
            },
            Genre.THRILLER: {
                "pacing": "fast",
                "emotional_range": "tense",
                "typical_conflicts": [ConflictType.PERSON_VS_PERSON],
                "typical_arcs": [ArcType.TESTING, ArcType.POSITIVE_CHANGE],
                "tone": "suspenseful"
            },
            Genre.MYSTERY: {
                "pacing": "methodical",
                "emotional_range": "intrigued",
                "typical_conflicts": [ConflictType.PERSON_VS_PERSON],
                "typical_arcs": [ArcType.FLAT, ArcType.POSITIVE_CHANGE],
                "tone": "investigative"
            },
            Genre.ROMANCE: {
                "pacing": "varies",
                "emotional_range": "romantic",
                "typical_conflicts": [ConflictType.PERSON_VS_SELF, ConflictType.PERSON_VS_SOCIETY],
                "typical_arcs": [ArcType.POSITIVE_CHANGE, ArcType.MATURATION],
                "tone": "romantic"
            }
        }
        return conventions.get(genre, conventions[Genre.DRAMA])

    def _register_default_generators(self):
        """Register default beat generators"""
        self.beat_generators["action"] = self._generate_action_beat
        self.beat_generators["reaction"] = self._generate_reaction_beat
        self.beat_generators["revelation"] = self._generate_revelation_beat
        self.beat_generators["decision"] = self._generate_decision_beat
        self.beat_generators["conflict"] = self._generate_conflict_beat

    def create_character(self, char_id: str, name: str, role: str,
                        want: str, need: str, flaw: str,
                        arc_type: ArcType = ArcType.POSITIVE_CHANGE) -> NarrativeCharacter:
        """Create a character with full psychological profile"""
        character = NarrativeCharacter(
            id=char_id,
            name=name,
            role=role,
            want=CharacterWant(description=want),
            need=CharacterNeed(description=need),
            flaw=CharacterFlaw(description=flaw),
            arc_type=arc_type
        )
        self.state.characters[char_id] = character
        return character

    def create_relationship(self, char1: str, char2: str,
                           rel_type: str, **kwargs) -> CharacterRelationship:
        """Create a relationship between characters"""
        rel_id = f"{char1}_{char2}"
        relationship = CharacterRelationship(
            character1=char1,
            character2=char2,
            relationship_type=rel_type,
            **kwargs
        )
        self.state.relationships[rel_id] = relationship
        return relationship

    def add_dramatic_question(self, question: str, stakes: str,
                             urgency: float = 0.5) -> DramaticQuestion:
        """Add a central dramatic question"""
        dq = DramaticQuestion(question=question, stakes=stakes, urgency=urgency)
        self.state.dramatic_questions.append(dq)
        return dq

    def add_theme(self, statement: str, question: str,
                 motifs: List[str] = None) -> Theme:
        """Add a thematic element"""
        theme = Theme(
            id=f"theme_{len(self.state.themes)}",
            statement=statement,
            question=question,
            motifs=motifs or []
        )
        self.state.themes.append(theme)
        return theme

    def transition_phase(self, new_phase: StoryPhase):
        """Transition to a new story phase"""
        self.state.current_phase = new_phase
        self.state.tension_curve.set_phase(new_phase)

    def generate_beat(self, beat_type: str, characters: List[str],
                     context: Dict[str, Any] = None) -> Beat:
        """Generate a story beat"""
        context = context or {}

        generator = self.beat_generators.get(beat_type, self._generate_generic_beat)
        beat = generator(characters, context)

        self.state.beat_count += 1
        beat.timestamp = self.state.beat_count
        self.state.timeline.append(beat)

        # Update tension
        self.state.tension_curve.update(self.state.beat_count)

        # Progress character arcs
        for char_id in characters:
            if char_id in self.state.characters:
                self._progress_arc(char_id, beat)

        return beat

    def _generate_generic_beat(self, characters: List[str],
                               context: Dict[str, Any]) -> Beat:
        """Generate a generic beat"""
        return Beat(
            id=f"beat_{self.state.beat_count}",
            description=context.get("description", "A moment passes"),
            beat_type="generic",
            characters_involved=characters
        )

    def _generate_action_beat(self, characters: List[str],
                              context: Dict[str, Any]) -> Beat:
        """Generate an action beat"""
        actor = characters[0] if characters else "unknown"
        action = context.get("action", "acts")

        return Beat(
            id=f"beat_{self.state.beat_count}",
            description=f"{actor} {action}",
            beat_type="action",
            characters_involved=characters,
            tension_change=context.get("tension_change", 0.1)
        )

    def _generate_reaction_beat(self, characters: List[str],
                                context: Dict[str, Any]) -> Beat:
        """Generate a reaction beat"""
        reactor = characters[0] if characters else "unknown"
        reaction = context.get("reaction", "reacts")

        return Beat(
            id=f"beat_{self.state.beat_count}",
            description=f"{reactor} {reaction}",
            beat_type="reaction",
            characters_involved=characters,
            emotional_beats={reactor: context.get("emotion", "affected")}
        )

    def _generate_revelation_beat(self, characters: List[str],
                                  context: Dict[str, Any]) -> Beat:
        """Generate a revelation beat"""
        revelation = context.get("revelation", "A truth is revealed")

        beat = Beat(
            id=f"beat_{self.state.beat_count}",
            description=revelation,
            beat_type="revelation",
            characters_involved=characters,
            tension_change=context.get("tension_change", 0.2)
        )

        # Update character knowledge
        for char_id in characters:
            if char_id in self.state.characters:
                self.state.characters[char_id].knowledge.add(revelation)

        return beat

    def _generate_decision_beat(self, characters: List[str],
                                context: Dict[str, Any]) -> Beat:
        """Generate a decision beat"""
        decider = characters[0] if characters else "unknown"
        decision = context.get("decision", "makes a choice")

        return Beat(
            id=f"beat_{self.state.beat_count}",
            description=f"{decider} {decision}",
            beat_type="decision",
            characters_involved=characters,
            tension_change=context.get("tension_change", 0.15)
        )

    def _generate_conflict_beat(self, characters: List[str],
                                context: Dict[str, Any]) -> Beat:
        """Generate a conflict beat"""
        conflict_desc = context.get("conflict", "Conflict erupts")

        return Beat(
            id=f"beat_{self.state.beat_count}",
            description=conflict_desc,
            beat_type="conflict",
            characters_involved=characters,
            tension_change=context.get("tension_change", 0.3)
        )

    def _progress_arc(self, char_id: str, beat: Beat):
        """Progress a character's arc based on a beat"""
        character = self.state.characters.get(char_id)
        if not character:
            return

        # Arc progression based on beat type and character state
        progression = beat.arc_progress.get(char_id, 0)

        if beat.beat_type in ["revelation", "decision"]:
            # These beats often advance arcs
            progression += 0.05
        elif beat.beat_type == "conflict":
            # Conflict accelerates arcs
            progression += 0.08

        # Modify by internal conflict
        internal_conflict = character.get_internal_conflict()
        progression *= (1 + internal_conflict * 0.5)

        character.arc_progress = min(1.0, character.arc_progress + progression)

        # Check for transformation moments
        thresholds = [0.25, 0.5, 0.75, 1.0]
        for threshold in thresholds:
            if character.arc_progress >= threshold:
                moment_key = f"transformation_{int(threshold * 100)}"
                if moment_key not in [m[:18] for m in character.transformation_moments]:
                    character.transformation_moments.append(
                        f"{moment_key}: {beat.description[:50]}"
                    )

    def evaluate_coherence(self) -> Dict[str, Any]:
        """Evaluate narrative coherence"""
        scores = {
            "character_consistency": self._eval_character_consistency(),
            "causal_chain": self._eval_causal_chain(),
            "tension_arc": self._eval_tension_arc(),
            "thematic_unity": self._eval_thematic_unity(),
            "pacing": self._eval_pacing()
        }

        overall = sum(scores.values()) / len(scores)

        return {
            "overall": overall,
            "details": scores,
            "suggestions": self._generate_coherence_suggestions(scores)
        }

    def _eval_character_consistency(self) -> float:
        """Evaluate character consistency"""
        if not self.state.characters:
            return 1.0

        scores = []
        for char in self.state.characters.values():
            # Check arc progress vs. beat count
            expected_progress = min(1.0, self.state.beat_count / 50)
            progress_diff = abs(char.arc_progress - expected_progress)
            scores.append(1.0 - progress_diff)

        return sum(scores) / len(scores)

    def _eval_causal_chain(self) -> float:
        """Evaluate causal chain integrity"""
        if len(self.state.timeline) < 2:
            return 1.0

        # Check if beats have causal connections
        connected = sum(1 for beat in self.state.timeline if beat.causes or beat.consequences)
        return connected / len(self.state.timeline)

    def _eval_tension_arc(self) -> float:
        """Evaluate tension arc appropriateness"""
        if not self.state.tension_curve.history:
            return 1.0

        # Check if tension follows expected pattern
        current = self.state.tension_curve.current_tension
        target = self.state.tension_curve.target_tension

        diff = abs(current - target)
        return 1.0 - min(1.0, diff)

    def _eval_thematic_unity(self) -> float:
        """Evaluate thematic unity"""
        if not self.state.themes:
            return 0.5  # Neutral if no themes defined

        # Check theme development
        developed = sum(1 for theme in self.state.themes if theme.development_beats)
        return developed / len(self.state.themes)

    def _eval_pacing(self) -> float:
        """Evaluate story pacing"""
        if not self.state.timeline:
            return 1.0

        # Check beat type distribution
        beat_types = [beat.beat_type for beat in self.state.timeline]
        type_counts = defaultdict(int)
        for bt in beat_types:
            type_counts[bt] += 1

        # Good pacing has variety
        if len(type_counts) < 2:
            return 0.5

        max_count = max(type_counts.values())
        dominance = max_count / len(beat_types)

        # If one type dominates too much, pacing is off
        return 1.0 - (dominance - 0.5) if dominance > 0.5 else 1.0

    def _generate_coherence_suggestions(self, scores: Dict[str, float]) -> List[str]:
        """Generate suggestions to improve coherence"""
        suggestions = []

        if scores["character_consistency"] < 0.7:
            suggestions.append("Consider adding more character development beats")

        if scores["causal_chain"] < 0.7:
            suggestions.append("Events feel disconnected - add more cause/effect relationships")

        if scores["tension_arc"] < 0.7:
            current = self.state.tension_curve.current_tension
            target = self.state.tension_curve.target_tension
            if current < target:
                suggestions.append("Tension is too low for this phase - add conflict")
            else:
                suggestions.append("Tension is too high - consider a moment of relief")

        if scores["thematic_unity"] < 0.7:
            suggestions.append("Themes need more development through the narrative")

        if scores["pacing"] < 0.7:
            suggestions.append("Add variety to beat types for better pacing")

        return suggestions

    def create_branch_point(self, description: str,
                           options: List[str]) -> Dict[str, Any]:
        """Create a branching point in the narrative"""
        branch = {
            "id": f"branch_{len(self.state.branch_points)}",
            "beat_number": self.state.beat_count,
            "description": description,
            "options": options,
            "chosen": None,
            "timestamp": datetime.now()
        }
        self.state.branch_points.append(branch)
        return branch

    def explore_counterfactual(self, branch_id: str, option: str) -> "StoryState":
        """Explore a counterfactual branch"""
        import copy

        # Create a copy of current state
        counterfactual = copy.deepcopy(self.state)
        counterfactual_key = f"{branch_id}_{option}"

        self.state.counterfactuals[counterfactual_key] = counterfactual

        return counterfactual

    def get_story_summary(self) -> Dict[str, Any]:
        """Get a summary of the current story state"""
        return {
            "genre": self.state.genre.value,
            "phase": self.state.current_phase.value,
            "act": self.state.current_act,
            "beat_count": self.state.beat_count,
            "characters": {
                char_id: {
                    "name": char.name,
                    "role": char.role,
                    "arc_progress": char.arc_progress,
                    "internal_conflict": char.get_internal_conflict()
                }
                for char_id, char in self.state.characters.items()
            },
            "tension": self.state.tension_curve.current_tension,
            "dramatic_questions": [
                {"question": dq.question, "answered": dq.answered}
                for dq in self.state.dramatic_questions
            ],
            "coherence": self.evaluate_coherence()["overall"]
        }


# ============================================================================
# STORY GENERATOR
# ============================================================================

class StoryGenerator:
    """
    Generates complete stories using the narrative engine.
    """

    def __init__(self, genre: Genre = Genre.DRAMA):
        self.engine = NarrativeEngine(genre)

    def generate_character_cast(self, num_characters: int = 4) -> List[NarrativeCharacter]:
        """Generate a cast of characters with complementary arcs"""
        roles = ["protagonist", "antagonist", "mentor", "ally", "trickster", "herald"]
        wants = [
            "power", "love", "freedom", "knowledge", "security", "recognition"
        ]
        needs = [
            "self-acceptance", "connection", "purpose", "truth", "growth", "forgiveness"
        ]
        flaws = [
            "pride", "fear", "anger", "distrust", "selfishness", "denial"
        ]

        characters = []
        for i in range(num_characters):
            role = roles[i] if i < len(roles) else "supporting"
            arc = ArcType.POSITIVE_CHANGE if i == 0 else random.choice(list(ArcType))

            char = self.engine.create_character(
                char_id=f"char_{i}",
                name=f"Character_{i}",
                role=role,
                want=random.choice(wants),
                need=random.choice(needs),
                flaw=random.choice(flaws),
                arc_type=arc
            )
            characters.append(char)

        # Create relationships between protagonist and others
        if len(characters) >= 2:
            for char in characters[1:]:
                rel_type = {
                    "antagonist": "enemy",
                    "mentor": "mentor",
                    "ally": "ally",
                    "trickster": "rival"
                }.get(char.role, "acquaintance")

                self.engine.create_relationship(
                    characters[0].id,
                    char.id,
                    rel_type
                )

        return characters

    def generate_story_structure(self) -> List[Act]:
        """Generate three-act structure"""
        acts = []

        # Act 1: Setup
        act1 = Act(
            id="act_1",
            act_number=1,
            goal="Establish world, character, and inciting incident"
        )
        acts.append(act1)

        # Act 2: Confrontation
        act2 = Act(
            id="act_2",
            act_number=2,
            goal="Escalate conflict, test protagonist, approach climax"
        )
        acts.append(act2)

        # Act 3: Resolution
        act3 = Act(
            id="act_3",
            act_number=3,
            goal="Climax and resolution"
        )
        acts.append(act3)

        self.engine.state.acts = acts
        return acts

    def generate_beats(self, num_beats: int = 20) -> List[Beat]:
        """Generate a sequence of beats"""
        beats = []
        beat_types = ["action", "reaction", "revelation", "decision", "conflict"]

        characters = list(self.engine.state.characters.keys())

        for i in range(num_beats):
            # Determine phase based on beat number
            progress = i / num_beats
            if progress < 0.25:
                self.engine.transition_phase(StoryPhase.SETUP)
            elif progress < 0.5:
                self.engine.transition_phase(StoryPhase.CONFRONTATION)
            elif progress < 0.75:
                self.engine.transition_phase(StoryPhase.ORDEAL)
            else:
                self.engine.transition_phase(StoryPhase.RESOLUTION)

            # Select beat type based on tension needs
            pacing = self.engine.state.tension_curve.get_pacing_suggestion()
            if pacing == "increase_tension":
                beat_type = random.choice(["conflict", "revelation", "action"])
            elif pacing == "release_tension":
                beat_type = random.choice(["reaction", "decision"])
            else:
                beat_type = random.choice(beat_types)

            # Select characters
            involved = random.sample(characters, min(2, len(characters)))

            beat = self.engine.generate_beat(
                beat_type,
                involved,
                {"tension_change": random.uniform(-0.1, 0.3)}
            )
            beats.append(beat)

        return beats


# Convenience functions
def create_narrative_engine(genre: Genre = Genre.DRAMA) -> NarrativeEngine:
    """Create a new narrative engine"""
    return NarrativeEngine(genre)


def create_story_generator(genre: Genre = Genre.DRAMA) -> StoryGenerator:
    """Create a new story generator"""
    return StoryGenerator(genre)


def demo_narrative():
    """Demonstrate narrative engine"""
    generator = create_story_generator(Genre.DRAMA)

    # Generate cast
    characters = generator.generate_character_cast(4)

    # Add dramatic question
    generator.engine.add_dramatic_question(
        "Will the protagonist overcome their flaw?",
        "Their relationships and goals hang in the balance"
    )

    # Add theme
    generator.engine.add_theme(
        "True strength comes from vulnerability",
        "Can one be both strong and vulnerable?",
        ["broken objects that become stronger", "scars"]
    )

    # Generate structure and beats
    generator.generate_story_structure()
    beats = generator.generate_beats(25)

    # Get summary
    summary = generator.engine.get_story_summary()

    print("=== Narrative Demo ===")
    print(f"Genre: {summary['genre']}")
    print(f"Characters: {len(summary['characters'])}")
    print(f"Beats: {summary['beat_count']}")
    print(f"Current tension: {summary['tension']:.2f}")
    print(f"Coherence: {summary['coherence']:.2f}")
    print(f"\nCharacter Arcs:")
    for char_id, char_info in summary['characters'].items():
        print(f"  {char_info['name']} ({char_info['role']}): {char_info['arc_progress']:.1%}")

    return generator


if __name__ == "__main__":
    demo_narrative()
