"""
Tests for the storytelling system.
"""

import pytest
import asyncio
from datetime import datetime


class TestStorytellingEngine:
    """Tests for the storytelling engine."""

    def test_engine_import(self):
        """Test that engine imports correctly."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryMode, StoryGenre,
            StorySession, StoryState, StoryBeat, StoryCharacter,
            NarrativePhase, TensionLevel,
        )
        assert StorytellingEngine is not None
        assert StoryConfig is not None
        assert StoryGenre is not None

    def test_story_config_creation(self):
        """Test creating story configuration."""
        from src.storytelling.engine import StoryConfig, StoryGenre, StoryMode

        config = StoryConfig(
            title="Test Story",
            genre=StoryGenre.DRAMA,
            mode=StoryMode.COLLABORATIVE,
            num_character_agents=3,
            target_length=20,
        )

        assert config.title == "Test Story"
        assert config.genre == StoryGenre.DRAMA
        assert config.mode == StoryMode.COLLABORATIVE
        assert config.num_character_agents == 3
        assert config.target_length == 20

    def test_story_config_to_dict(self):
        """Test config serialization."""
        from src.storytelling.engine import StoryConfig, StoryGenre, StoryMode

        config = StoryConfig(
            title="Test",
            genre=StoryGenre.THRILLER,
            themes=["suspense", "betrayal"],
        )

        d = config.to_dict()
        assert d["title"] == "Test"
        assert d["genre"] == "thriller"
        assert d["themes"] == ["suspense", "betrayal"]

    def test_engine_creation(self):
        """Test creating storytelling engine."""
        from src.storytelling.engine import StorytellingEngine

        engine = StorytellingEngine()
        assert engine is not None
        assert len(engine.sessions) == 0
        assert len(engine.genre_conventions) > 0

    def test_create_session(self):
        """Test creating a story session."""
        from src.storytelling.engine import StorytellingEngine, StoryConfig, StoryGenre

        engine = StorytellingEngine()
        config = StoryConfig(title="Test Session", genre=StoryGenre.MYSTERY)

        session = engine.create_session(config)

        assert session is not None
        assert session.id.startswith("STORY-")
        assert session.state.config.title == "Test Session"
        assert "main" in session.state.plot_threads

    def test_add_character(self):
        """Test adding characters to story."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryCharacter,
        )

        engine = StorytellingEngine()
        session = engine.create_session(StoryConfig(title="Test"))

        char = StoryCharacter(
            id="hero",
            name="The Hero",
            role="protagonist",
            arc_type="growth",
            motivations=["justice", "redemption"],
        )

        result = engine.add_character(session.id, char)

        assert result is True
        assert "hero" in session.state.characters
        assert session.state.characters["hero"].name == "The Hero"

    def test_add_beat(self):
        """Test adding story beats."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryBeat,
        )

        engine = StorytellingEngine()
        session = engine.create_session(StoryConfig(title="Test"))

        beat = StoryBeat(
            id="beat_001",
            content="The story begins with a dark and stormy night.",
            beat_type="description",
            author_agent="narrator",
        )

        result = engine.add_beat(session.id, beat)

        assert result is True
        assert len(session.state.beats) == 1
        assert session.state.word_count > 0

    def test_narrative_phase_progression(self):
        """Test that narrative phase progresses with beats."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryBeat, NarrativePhase,
        )

        engine = StorytellingEngine()
        config = StoryConfig(title="Test", target_length=10)
        session = engine.create_session(config)

        # Add beats to progress through phases
        for i in range(5):
            beat = StoryBeat(
                id=f"beat_{i}",
                content=f"Story beat number {i}",
                beat_type="action",
                author_agent="test",
            )
            engine.add_beat(session.id, beat)

        # Should have progressed past setup
        assert session.state.current_phase != NarrativePhase.SETUP

    def test_story_render_text(self):
        """Test rendering story to text."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryBeat,
        )

        engine = StorytellingEngine()
        session = engine.create_session(StoryConfig(title="Render Test"))

        engine.add_beat(session.id, StoryBeat(
            id="b1",
            content="Once upon a time...",
            beat_type="description",
            author_agent="narrator",
        ))

        text = engine.render_story(session, format="text")

        assert "Render Test" in text
        assert "Once upon a time" in text

    def test_coherence_evaluation(self):
        """Test story coherence evaluation."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryBeat,
        )

        engine = StorytellingEngine()
        session = engine.create_session(StoryConfig(title="Test"))

        # Add some beats
        for i in range(3):
            engine.add_beat(session.id, StoryBeat(
                id=f"b{i}",
                content=f"Content {i}",
                beat_type="action",
                author_agent="test",
            ))

        coherence = engine.evaluate_coherence(session)

        assert "overall" in coherence
        assert isinstance(coherence["overall"], float)
        assert 0 <= coherence["overall"] <= 1


class TestStorytellingAgents:
    """Tests for storytelling agents."""

    def test_agents_import(self):
        """Test that agents import correctly."""
        from src.storytelling.agents import (
            StoryAgent, NarratorAgent, CharacterAgent,
            DirectorAgent, CriticAgent, StoryAgentFactory,
            AgentPersonality, AgentRole,
        )
        assert NarratorAgent is not None
        assert CharacterAgent is not None
        assert DirectorAgent is not None

    def test_narrator_creation(self):
        """Test creating narrator agent."""
        from src.storytelling.agents import NarratorAgent, AgentRole

        narrator = NarratorAgent()

        assert narrator.name == "The Narrator"
        assert narrator.role == AgentRole.NARRATOR
        assert narrator.contributions == 0

    def test_character_agent_creation(self):
        """Test creating character agent."""
        from src.storytelling.agents import CharacterAgent, AgentRole
        from src.storytelling.engine import StoryCharacter

        char = StoryCharacter(
            id="test_char",
            name="Test Character",
            role="protagonist",
        )

        agent = CharacterAgent(char)

        assert agent.character.name == "Test Character"
        assert agent.role == AgentRole.CHARACTER

    def test_director_creation(self):
        """Test creating director agent."""
        from src.storytelling.agents import DirectorAgent, AgentRole

        director = DirectorAgent()

        assert director.name == "The Director"
        assert director.role == AgentRole.DIRECTOR

    def test_agent_factory(self):
        """Test agent factory."""
        from src.storytelling.agents import StoryAgentFactory
        from src.storytelling.engine import StoryCharacter

        characters = [
            StoryCharacter(id="c1", name="Char 1", role="protagonist"),
            StoryCharacter(id="c2", name="Char 2", role="antagonist"),
        ]

        agents = StoryAgentFactory.create_ensemble(
            characters,
            include_narrator=True,
            include_director=True,
            include_critic=True,
        )

        assert "narrator" in agents
        assert "director" in agents
        assert "critic" in agents
        assert "char_c1" in agents
        assert "char_c2" in agents

    def test_agent_personality(self):
        """Test agent personality configuration."""
        from src.storytelling.agents import NarratorAgent, AgentPersonality

        personality = AgentPersonality(
            creativity=0.9,
            verbosity=0.3,
            drama_preference=0.8,
        )

        narrator = NarratorAgent(personality=personality)

        assert narrator.personality.creativity == 0.9
        assert narrator.personality.verbosity == 0.3

    @pytest.mark.asyncio
    async def test_narrator_generate_contribution(self):
        """Test narrator generating contribution."""
        from src.storytelling.agents import NarratorAgent
        from src.storytelling.engine import StorytellingEngine, StoryConfig

        engine = StorytellingEngine()
        session = engine.create_session(StoryConfig(title="Test"))

        narrator = NarratorAgent()
        context = engine.generate_prompt_context(session)

        beat = await narrator.generate_contribution(session, context)

        assert beat is not None
        assert beat.author_agent == "narrator"
        assert beat.content is not None


class TestCollaborativeStory:
    """Tests for collaborative storytelling."""

    def test_collaborative_import(self):
        """Test that collaborative module imports correctly."""
        from src.storytelling.collaborative import (
            CollaborativeStory, CollaborativeStoryConfig,
            StoryContribution, StoryRound, ContributionType,
        )
        assert CollaborativeStory is not None
        assert StoryContribution is not None

    def test_collaborative_config(self):
        """Test collaborative story configuration."""
        from src.storytelling.collaborative import CollaborativeStoryConfig
        from src.storytelling.engine import StoryConfig, StoryGenre

        story_config = StoryConfig(
            title="Collab Test",
            genre=StoryGenre.DRAMA,
        )

        config = CollaborativeStoryConfig(
            story_config=story_config,
            max_rounds=10,
            enable_critique=True,
        )

        assert config.story_config.title == "Collab Test"
        assert config.max_rounds == 10
        assert config.enable_critique is True

    def test_collaborative_story_creation(self):
        """Test creating collaborative story."""
        from src.storytelling.collaborative import (
            CollaborativeStory, CollaborativeStoryConfig,
        )
        from src.storytelling.engine import StoryConfig

        config = CollaborativeStoryConfig(
            story_config=StoryConfig(title="Test"),
            max_rounds=5,
        )

        story = CollaborativeStory(config)

        assert story.session is None  # Not initialized yet
        assert story.is_running is False

    def test_collaborative_story_initialize(self):
        """Test initializing collaborative story."""
        from src.storytelling.collaborative import (
            CollaborativeStory, CollaborativeStoryConfig,
        )
        from src.storytelling.engine import StoryConfig

        config = CollaborativeStoryConfig(
            story_config=StoryConfig(title="Init Test", num_character_agents=2),
            max_rounds=5,
        )

        story = CollaborativeStory(config)
        session = story.initialize()

        assert session is not None
        assert session.status == "active"
        assert len(story.agents) > 0
        assert len(story.turn_order) > 0

    def test_create_collaborative_story_helper(self):
        """Test the helper function for creating stories."""
        from src.storytelling.collaborative import create_collaborative_story

        story = create_collaborative_story(
            title="Helper Test",
            genre="thriller",
            num_characters=3,
            max_rounds=10,
        )

        assert story is not None
        assert story.config.story_config.title == "Helper Test"
        assert story.config.story_config.genre.value == "thriller"


class TestStoryContribution:
    """Tests for story contributions."""

    def test_contribution_creation(self):
        """Test creating story contribution."""
        from src.storytelling.collaborative import (
            StoryContribution, ContributionType,
        )

        contrib = StoryContribution(
            id="contrib_1",
            content="The hero stepped forward...",
            contribution_type=ContributionType.NARRATIVE,
            agent_id="narrator",
            agent_name="The Narrator",
            round_number=1,
        )

        assert contrib.id == "contrib_1"
        assert contrib.word_count > 0
        assert contrib.approved is True

    def test_story_round(self):
        """Test story round tracking."""
        from src.storytelling.collaborative import StoryRound
        from src.storytelling.engine import NarrativePhase, TensionLevel

        round_obj = StoryRound(
            round_number=1,
            phase=NarrativePhase.RISING_ACTION,
            tension=TensionLevel.MODERATE,
        )

        assert round_obj.round_number == 1
        assert round_obj.phase == NarrativePhase.RISING_ACTION


class TestGenreConventions:
    """Tests for genre conventions."""

    def test_all_genres_have_conventions(self):
        """Test that all genres have defined conventions."""
        from src.storytelling.engine import StorytellingEngine, StoryGenre

        engine = StorytellingEngine()

        test_genres = [
            StoryGenre.DRAMA,
            StoryGenre.COMEDY,
            StoryGenre.THRILLER,
            StoryGenre.MYSTERY,
            StoryGenre.ROMANCE,
            StoryGenre.SCIFI,
            StoryGenre.FANTASY,
            StoryGenre.HORROR,
        ]

        for genre in test_genres:
            assert genre in engine.genre_conventions
            conv = engine.genre_conventions[genre]
            assert "pacing" in conv
            assert "focus" in conv
            assert "tension_multiplier" in conv

    def test_tension_curves_defined(self):
        """Test that tension curves are defined."""
        from src.storytelling.engine import StorytellingEngine

        engine = StorytellingEngine()

        curves = ["classic", "slow_burn", "thriller", "wave"]
        for curve in curves:
            assert curve in engine.TENSION_CURVES
            assert len(engine.TENSION_CURVES[curve]) > 0


class TestIntegration:
    """Integration tests for storytelling system."""

    @pytest.mark.asyncio
    async def test_full_story_flow(self):
        """Test complete story generation flow."""
        from src.storytelling.collaborative import create_collaborative_story
        from src.storytelling.engine import StoryCharacter

        story = create_collaborative_story(
            title="Integration Test",
            genre="drama",
            num_characters=2,
            max_rounds=3,
        )

        characters = [
            StoryCharacter(
                id="alice",
                name="Alice",
                role="protagonist",
                motivations=["truth"],
            ),
            StoryCharacter(
                id="bob",
                name="Bob",
                role="antagonist",
                motivations=["power"],
            ),
        ]

        story.initialize(characters)

        # Run limited story
        story.config.max_rounds = 2
        state = await story.run()

        assert state is not None
        assert len(state.beats) > 0
        assert state.word_count > 0

    def test_story_output_formats(self):
        """Test different output formats."""
        from src.storytelling.engine import (
            StorytellingEngine, StoryConfig, StoryBeat, StoryGenre,
        )

        engine = StorytellingEngine()
        session = engine.create_session(StoryConfig(
            title="Format Test",
            genre=StoryGenre.DRAMA,
        ))

        engine.add_beat(session.id, StoryBeat(
            id="b1",
            content="The end is just the beginning.",
            beat_type="description",
            author_agent="narrator",
        ))

        # Test text format
        text = engine.render_story(session, format="text")
        assert "Format Test" in text
        assert "The end is just the beginning" in text

        # Test JSON format
        json_str = engine.render_story(session, format="json")
        assert "Format Test" in json_str
        assert "beats" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
