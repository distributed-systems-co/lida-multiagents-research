"""
Dinner Party Simulation Engine

Interactive dinner party simulations with:
- Famous personas from all domains
- Human intervention hooks (stop, edit, guide)
- Comprehensive data collection
- Dynamic conversation flow
- Relationship tracking
"""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Awaitable

from .famous_personas import ALL_PERSONAS, get_persona, PersonaDomain


# =============================================================================
# Core Types
# =============================================================================

class ConversationTopic(Enum):
    """Topics that can come up at dinner."""
    SMALL_TALK = "small_talk"
    POLITICS = "politics"
    TECHNOLOGY = "technology"
    ARTS_CULTURE = "arts_culture"
    SPORTS = "sports"
    PHILOSOPHY = "philosophy"
    GOSSIP = "gossip"
    BUSINESS = "business"
    PERSONAL_STORIES = "personal_stories"
    CONTROVERSIAL = "controversial"
    FOOD_WINE = "food_wine"


class InteractionType(Enum):
    """Types of dinner party interactions."""
    STATEMENT = "statement"
    QUESTION = "question"
    RESPONSE = "response"
    JOKE = "joke"
    STORY = "story"
    TOAST = "toast"
    ASIDE = "aside"  # Whispered to neighbor
    DEBATE = "debate"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    COMPLIMENT = "compliment"
    SUBTLE_DIG = "subtle_dig"


class GuestMood(Enum):
    """Guest emotional states during dinner."""
    ENJOYING = "enjoying"
    BORED = "bored"
    ENGAGED = "engaged"
    UNCOMFORTABLE = "uncomfortable"
    AMUSED = "amused"
    ANNOYED = "annoyed"
    CHARMED = "charmed"
    COMPETITIVE = "competitive"
    REFLECTIVE = "reflective"
    TIPSY = "tipsy"


class HookType(Enum):
    """Types of human intervention hooks."""
    BEFORE_TURN = "before_turn"        # Before a guest speaks
    AFTER_TURN = "after_turn"          # After a guest speaks
    TOPIC_CHANGE = "topic_change"      # When topic is about to change
    TENSION_RISING = "tension_rising"  # When conflict is brewing
    AWKWARD_SILENCE = "awkward_silence"
    EDIT_MESSAGE = "edit_message"      # Human wants to edit what was said
    INJECT_EVENT = "inject_event"      # Human injects an event
    PAUSE = "pause"                    # General pause
    GUIDE = "guide"                    # Provide guidance to a guest


@dataclass
class Utterance:
    """A single thing said at the dinner party."""
    id: str
    speaker_id: str
    speaker_name: str
    content: str
    interaction_type: InteractionType
    topic: ConversationTopic
    target_id: Optional[str] = None
    target_name: Optional[str] = None
    responding_to: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Dynamics
    humor_level: float = 0.0  # -1 to 1
    controversy_level: float = 0.0  # 0 to 1
    intimacy_level: float = 0.0  # 0 to 1 (how personal)

    # Tracking
    reactions: Dict[str, str] = field(default_factory=dict)  # guest_id -> reaction
    edited: bool = False
    original_content: Optional[str] = None


@dataclass
class GuestState:
    """State of a dinner party guest."""
    id: str
    persona_id: str
    name: str
    seat_position: int

    # Current state
    mood: GuestMood = GuestMood.ENJOYING
    energy: float = 1.0  # 0 to 1
    intoxication: float = 0.0  # 0 to 1
    engagement: float = 0.7  # 0 to 1

    # Social dynamics
    comfort_with: Dict[str, float] = field(default_factory=dict)  # guest_id -> comfort level
    interest_in: Dict[str, float] = field(default_factory=dict)  # guest_id -> interest level
    tension_with: Dict[str, float] = field(default_factory=dict)  # guest_id -> tension level

    # Conversation
    utterances_made: List[str] = field(default_factory=list)
    topics_raised: List[ConversationTopic] = field(default_factory=list)
    times_interrupted: int = 0
    times_agreed_with: int = 0
    times_disagreed_with: int = 0

    # Impressions formed
    impressions: Dict[str, str] = field(default_factory=dict)  # guest_id -> impression text


@dataclass
class DinnerEvent:
    """Special events during dinner."""
    id: str
    event_type: str  # "wine_spill", "phone_rings", "toast", "food_arrives", etc.
    description: str
    triggered_by: Optional[str] = None
    affects: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HookContext:
    """Context passed to hook callbacks."""
    hook_type: HookType
    current_speaker: Optional[str]
    next_speaker: Optional[str]
    current_topic: ConversationTopic
    tension_level: float
    recent_utterances: List[Utterance]
    guests: Dict[str, GuestState]

    # For edit hooks
    proposed_content: Optional[str] = None


@dataclass
class HookResult:
    """Result from a hook callback."""
    action: str  # "continue", "skip", "edit", "inject", "pause", "stop"
    edited_content: Optional[str] = None
    injected_event: Optional[str] = None
    guidance: Optional[str] = None
    skip_speaker: bool = False
    new_topic: Optional[ConversationTopic] = None


@dataclass
class DinnerPartyState:
    """Complete state of a dinner party simulation."""
    id: str
    name: str
    setting: str  # "intimate_loft", "formal_mansion", "rooftop_penthouse", etc.

    # Guests
    guests: Dict[str, GuestState] = field(default_factory=dict)
    seating_order: List[str] = field(default_factory=list)
    host_id: Optional[str] = None

    # Conversation
    current_topic: ConversationTopic = ConversationTopic.SMALL_TALK
    current_speaker: Optional[str] = None
    conversation_flow: List[str] = field(default_factory=list)  # Utterance IDs
    all_utterances: Dict[str, Utterance] = field(default_factory=dict)

    # Events
    events: List[DinnerEvent] = field(default_factory=list)

    # Progress
    course: int = 1  # 1=appetizer, 2=main, 3=dessert, 4=drinks
    round: int = 0
    started_at: datetime = field(default_factory=datetime.now)

    # Dynamics
    overall_energy: float = 0.7
    overall_tension: float = 0.1
    topics_exhausted: Set[ConversationTopic] = field(default_factory=set)


# =============================================================================
# Data Collection
# =============================================================================

@dataclass
class DataCollector:
    """Comprehensive data collection for dinner party simulations."""

    simulation_id: str
    started_at: datetime = field(default_factory=datetime.now)

    # Raw data
    all_utterances: List[Dict[str, Any]] = field(default_factory=list)
    all_events: List[Dict[str, Any]] = field(default_factory=list)
    guest_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    hook_interactions: List[Dict[str, Any]] = field(default_factory=list)

    # Aggregated metrics
    topic_distribution: Dict[str, int] = field(default_factory=dict)
    speaker_distribution: Dict[str, int] = field(default_factory=dict)
    interaction_types: Dict[str, int] = field(default_factory=dict)
    mood_changes: List[Dict[str, Any]] = field(default_factory=list)
    relationship_changes: List[Dict[str, Any]] = field(default_factory=list)

    def record_utterance(self, utterance: Utterance):
        """Record an utterance."""
        self.all_utterances.append({
            "id": utterance.id,
            "speaker_id": utterance.speaker_id,
            "speaker_name": utterance.speaker_name,
            "content": utterance.content,
            "interaction_type": utterance.interaction_type.value,
            "topic": utterance.topic.value,
            "target_id": utterance.target_id,
            "target_name": utterance.target_name,
            "responding_to": utterance.responding_to,
            "timestamp": utterance.timestamp.isoformat(),
            "humor_level": utterance.humor_level,
            "controversy_level": utterance.controversy_level,
            "intimacy_level": utterance.intimacy_level,
            "edited": utterance.edited,
            "original_content": utterance.original_content,
        })

        # Update distributions
        topic_key = utterance.topic.value
        self.topic_distribution[topic_key] = self.topic_distribution.get(topic_key, 0) + 1

        speaker_key = utterance.speaker_id
        self.speaker_distribution[speaker_key] = self.speaker_distribution.get(speaker_key, 0) + 1

        type_key = utterance.interaction_type.value
        self.interaction_types[type_key] = self.interaction_types.get(type_key, 0) + 1

    def record_event(self, event: DinnerEvent):
        """Record a dinner event."""
        self.all_events.append({
            "id": event.id,
            "event_type": event.event_type,
            "description": event.description,
            "triggered_by": event.triggered_by,
            "affects": event.affects,
            "timestamp": event.timestamp.isoformat(),
        })

    def record_guest_snapshot(self, guest: GuestState, round_num: int):
        """Record a snapshot of guest state."""
        self.guest_snapshots.append({
            "guest_id": guest.id,
            "name": guest.name,
            "round": round_num,
            "mood": guest.mood.value,
            "energy": guest.energy,
            "intoxication": guest.intoxication,
            "engagement": guest.engagement,
            "comfort_with": dict(guest.comfort_with),
            "tension_with": dict(guest.tension_with),
            "timestamp": datetime.now().isoformat(),
        })

    def record_hook_interaction(self, hook_type: HookType, result: HookResult, context: Dict[str, Any]):
        """Record a human intervention."""
        self.hook_interactions.append({
            "hook_type": hook_type.value,
            "action": result.action,
            "edited_content": result.edited_content,
            "injected_event": result.injected_event,
            "guidance": result.guidance,
            "context": context,
            "timestamp": datetime.now().isoformat(),
        })

    def record_mood_change(self, guest_id: str, old_mood: GuestMood, new_mood: GuestMood, cause: str):
        """Record a mood change."""
        self.mood_changes.append({
            "guest_id": guest_id,
            "old_mood": old_mood.value,
            "new_mood": new_mood.value,
            "cause": cause,
            "timestamp": datetime.now().isoformat(),
        })

    def record_relationship_change(self, guest_a: str, guest_b: str, dimension: str, old_val: float, new_val: float):
        """Record a relationship change."""
        self.relationship_changes.append({
            "guest_a": guest_a,
            "guest_b": guest_b,
            "dimension": dimension,
            "old_value": old_val,
            "new_value": new_val,
            "timestamp": datetime.now().isoformat(),
        })

    def export_json(self, path: str):
        """Export all data to JSON."""
        data = {
            "simulation_id": self.simulation_id,
            "started_at": self.started_at.isoformat(),
            "exported_at": datetime.now().isoformat(),
            "utterances": self.all_utterances,
            "events": self.all_events,
            "guest_snapshots": self.guest_snapshots,
            "hook_interactions": self.hook_interactions,
            "metrics": {
                "topic_distribution": self.topic_distribution,
                "speaker_distribution": self.speaker_distribution,
                "interaction_types": self.interaction_types,
                "mood_changes": self.mood_changes,
                "relationship_changes": self.relationship_changes,
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected data."""
        return {
            "total_utterances": len(self.all_utterances),
            "total_events": len(self.all_events),
            "human_interventions": len(self.hook_interactions),
            "topics_discussed": list(self.topic_distribution.keys()),
            "most_talkative": max(self.speaker_distribution.items(), key=lambda x: x[1])[0] if self.speaker_distribution else None,
            "mood_changes": len(self.mood_changes),
            "relationship_changes": len(self.relationship_changes),
        }


# =============================================================================
# Dinner Party Engine
# =============================================================================

class DinnerPartyEngine:
    """
    Interactive dinner party simulation engine.

    Features:
    - Rich persona interactions
    - Human intervention hooks
    - Comprehensive data collection
    - Dynamic conversation flow
    - Relationship evolution
    """

    def __init__(
        self,
        party_name: str,
        guest_ids: List[str],
        setting: str = "elegant_dining_room",
        host_id: Optional[str] = None,
    ):
        self.party_name = party_name
        self.setting = setting

        # Initialize state
        self.state = DinnerPartyState(
            id=str(uuid.uuid4())[:8],
            name=party_name,
            setting=setting,
            host_id=host_id or (guest_ids[0] if guest_ids else None),
        )

        # Initialize guests
        for i, guest_id in enumerate(guest_ids):
            if guest_id in ALL_PERSONAS:
                self._init_guest(guest_id, seat_position=i)

        self.state.seating_order = list(self.state.guests.keys())

        # Data collection
        self.collector = DataCollector(simulation_id=self.state.id)

        # Hooks
        self.hooks: Dict[HookType, List[Callable[[HookContext], Awaitable[HookResult]]]] = {
            hook_type: [] for hook_type in HookType
        }

        # Control
        self.paused = False
        self.stopped = False

        # Callbacks
        self.on_utterance: Optional[Callable[[Utterance], None]] = None
        self.on_event: Optional[Callable[[DinnerEvent], None]] = None
        self.on_mood_change: Optional[Callable[[str, GuestMood, GuestMood], None]] = None
        self.on_topic_change: Optional[Callable[[ConversationTopic, ConversationTopic], None]] = None

    def _init_guest(self, persona_id: str, seat_position: int):
        """Initialize a guest from persona."""
        persona = get_persona(persona_id)

        guest = GuestState(
            id=persona_id,
            persona_id=persona_id,
            name=persona.get("name", persona_id),
            seat_position=seat_position,
        )

        # Initialize relationships with other guests
        persona_relationships = persona.get("relationships", {})
        for other_id in self.state.guests:
            # Check if there's a predefined relationship
            if other_id in persona_relationships:
                rel_type = persona_relationships[other_id]
                if "friend" in rel_type or "ally" in rel_type:
                    guest.comfort_with[other_id] = 0.7
                    guest.interest_in[other_id] = 0.6
                elif "hostile" in rel_type or "rival" in rel_type:
                    guest.comfort_with[other_id] = 0.3
                    guest.tension_with[other_id] = 0.4
                else:
                    guest.comfort_with[other_id] = 0.5
            else:
                guest.comfort_with[other_id] = 0.5
                guest.interest_in[other_id] = 0.5

        self.state.guests[persona_id] = guest

        # Update other guests' relationships to this guest
        for other_id, other_guest in self.state.guests.items():
            if other_id != persona_id:
                if persona_id not in other_guest.comfort_with:
                    other_guest.comfort_with[persona_id] = 0.5
                    other_guest.interest_in[persona_id] = 0.5

    # === Hook Management ===

    def add_hook(self, hook_type: HookType, callback: Callable[[HookContext], Awaitable[HookResult]]):
        """Add a hook callback."""
        self.hooks[hook_type].append(callback)

    def remove_hook(self, hook_type: HookType, callback: Callable):
        """Remove a hook callback."""
        if callback in self.hooks[hook_type]:
            self.hooks[hook_type].remove(callback)

    async def _trigger_hook(self, hook_type: HookType, context: HookContext) -> HookResult:
        """Trigger all hooks of a type and return combined result."""
        result = HookResult(action="continue")

        for callback in self.hooks[hook_type]:
            try:
                hook_result = await callback(context)

                # Record the interaction
                self.collector.record_hook_interaction(
                    hook_type,
                    hook_result,
                    {
                        "current_speaker": context.current_speaker,
                        "current_topic": context.current_topic.value,
                        "tension_level": context.tension_level,
                    }
                )

                # Handle result
                if hook_result.action == "stop":
                    self.stopped = True
                    return hook_result
                elif hook_result.action == "pause":
                    self.paused = True
                    return hook_result
                elif hook_result.action in ["edit", "inject", "skip"]:
                    return hook_result

            except Exception as e:
                print(f"Hook error: {e}")

        return result

    # === Simulation Control ===

    def pause(self):
        """Pause the simulation."""
        self.paused = True

    def resume(self):
        """Resume the simulation."""
        self.paused = False

    def stop(self):
        """Stop the simulation."""
        self.stopped = True

    # === Core Simulation ===

    async def run_round(self) -> List[Utterance]:
        """Run a single round of conversation."""
        if self.stopped:
            return []

        while self.paused:
            await asyncio.sleep(0.1)

        self.state.round += 1
        round_utterances = []

        # Take snapshots of all guests
        for guest in self.state.guests.values():
            self.collector.record_guest_snapshot(guest, self.state.round)

        # Maybe change topic
        if random.random() < 0.3 and self.state.round > 1:
            await self._maybe_change_topic()

        # Each guest gets a chance to speak
        for guest_id in self.state.seating_order:
            if self.stopped:
                break

            guest = self.state.guests[guest_id]

            # Check engagement - might skip if bored
            if guest.engagement < 0.3 and random.random() > guest.engagement:
                continue

            # Before turn hook
            context = self._build_hook_context(HookType.BEFORE_TURN, guest_id)
            result = await self._trigger_hook(HookType.BEFORE_TURN, context)

            if result.action == "skip" or result.skip_speaker:
                continue

            # Generate utterance
            utterance = await self._generate_utterance(guest_id)

            if utterance:
                # Edit hook check
                if result.action == "edit" and result.edited_content:
                    utterance.original_content = utterance.content
                    utterance.content = result.edited_content
                    utterance.edited = True

                # Store and notify
                self.state.all_utterances[utterance.id] = utterance
                self.state.conversation_flow.append(utterance.id)
                guest.utterances_made.append(utterance.id)

                self.collector.record_utterance(utterance)

                if self.on_utterance:
                    self.on_utterance(utterance)

                round_utterances.append(utterance)

                # Process effects on other guests
                await self._process_utterance_effects(utterance)

                # After turn hook
                after_context = self._build_hook_context(HookType.AFTER_TURN, guest_id)
                await self._trigger_hook(HookType.AFTER_TURN, after_context)

        # Check for special events
        await self._check_events()

        # Update overall dynamics
        self._update_party_dynamics()

        return round_utterances

    async def _generate_utterance(self, speaker_id: str) -> Optional[Utterance]:
        """Generate an utterance for a guest."""
        guest = self.state.guests[speaker_id]
        persona = get_persona(speaker_id)

        self.state.current_speaker = speaker_id

        # Determine interaction type based on context
        interaction_type = self._choose_interaction_type(guest)

        # Choose target (if any)
        target_id = self._choose_target(speaker_id)
        target = self.state.guests.get(target_id) if target_id else None

        # Generate content based on persona
        content = self._generate_content(guest, persona, interaction_type, target_id)

        # Calculate dynamics
        humor = self._calculate_humor_level(content, persona)
        controversy = self._calculate_controversy_level(content, self.state.current_topic)
        intimacy = self._calculate_intimacy_level(content, guest)

        utterance = Utterance(
            id=f"utt_{len(self.state.all_utterances):04d}",
            speaker_id=speaker_id,
            speaker_name=guest.name,
            content=content,
            interaction_type=interaction_type,
            topic=self.state.current_topic,
            target_id=target_id,
            target_name=target.name if target else None,
            responding_to=self.state.conversation_flow[-1] if self.state.conversation_flow else None,
            humor_level=humor,
            controversy_level=controversy,
            intimacy_level=intimacy,
        )

        return utterance

    def _choose_interaction_type(self, guest: GuestState) -> InteractionType:
        """Choose what type of interaction based on guest state."""
        if guest.mood == GuestMood.AMUSED:
            return random.choice([InteractionType.JOKE, InteractionType.STORY, InteractionType.STATEMENT])
        elif guest.mood == GuestMood.COMPETITIVE:
            return random.choice([InteractionType.DISAGREEMENT, InteractionType.DEBATE, InteractionType.QUESTION])
        elif guest.mood == GuestMood.CHARMED:
            return random.choice([InteractionType.COMPLIMENT, InteractionType.AGREEMENT, InteractionType.STORY])
        elif guest.mood == GuestMood.BORED:
            return random.choice([InteractionType.QUESTION, InteractionType.ASIDE])
        elif guest.mood == GuestMood.TIPSY:
            return random.choice([InteractionType.JOKE, InteractionType.STORY, InteractionType.TOAST])
        else:
            return random.choice(list(InteractionType))

    def _choose_target(self, speaker_id: str) -> Optional[str]:
        """Choose who to address."""
        guest = self.state.guests[speaker_id]

        # Prefer neighbors at table
        seat_pos = guest.seat_position
        neighbors = []
        for other_id, other in self.state.guests.items():
            if other_id != speaker_id:
                if abs(other.seat_position - seat_pos) <= 1:
                    neighbors.append(other_id)

        # Weight by interest and comfort
        candidates = list(self.state.guests.keys())
        candidates.remove(speaker_id)

        if not candidates:
            return None

        # Simple weighted random
        weights = []
        for cid in candidates:
            weight = 0.5
            weight += guest.interest_in.get(cid, 0.5) * 0.3
            if cid in neighbors:
                weight += 0.2
            if guest.tension_with.get(cid, 0) > 0.5:
                weight += 0.1  # Tension is interesting
            weights.append(weight)

        return random.choices(candidates, weights=weights)[0]

    def _generate_content(
        self,
        guest: GuestState,
        persona: Dict[str, Any],
        interaction_type: InteractionType,
        target_id: Optional[str]
    ) -> str:
        """Generate content based on persona and context."""
        # Get persona characteristics
        speaking_style = persona.get("speaking_style", "conversational")
        small_talk_topics = persona.get("small_talk_topics", [])
        dinner_behavior = persona.get("dinner_party_behavior", "engages normally")
        signature_phrases = persona.get("signature_phrases", [])

        # Build content based on type and topic
        topic = self.state.current_topic

        if interaction_type == InteractionType.JOKE:
            templates = [
                f"You know what's funny about {topic.value.replace('_', ' ')}...",
                f"This reminds me of something hilarious - ",
                f"Speaking of which, did you hear about...",
            ]
            if random.random() < 0.3 and signature_phrases:
                return random.choice(signature_phrases)
            return random.choice(templates)

        elif interaction_type == InteractionType.STORY:
            if small_talk_topics:
                topic_text = random.choice(small_talk_topics)
                return f"That reminds me of {topic_text}..."
            return f"Let me tell you about something that happened..."

        elif interaction_type == InteractionType.QUESTION:
            target = self.state.guests.get(target_id)
            if target:
                target_persona = get_persona(target_id)
                target_topics = target_persona.get("small_talk_topics", [])
                if target_topics:
                    return f"So {target.name}, tell me about {random.choice(target_topics)}?"
            return "What does everyone think about that?"

        elif interaction_type == InteractionType.TOAST:
            return f"I'd like to raise a glass to {random.choice(['good company', 'this wonderful evening', 'our host'])}!"

        elif interaction_type == InteractionType.AGREEMENT:
            return random.choice([
                "Exactly! That's precisely how I see it.",
                "I couldn't agree more.",
                "You've hit the nail on the head.",
            ])

        elif interaction_type == InteractionType.DISAGREEMENT:
            return random.choice([
                "Well, I see it quite differently...",
                "I'm not so sure about that.",
                "Interesting perspective, but have you considered...",
            ])

        elif interaction_type == InteractionType.COMPLIMENT:
            target = self.state.guests.get(target_id)
            if target:
                return f"I have to say, {target.name}, I've always admired your work on..."
            return "The food is absolutely exquisite tonight."

        elif interaction_type == InteractionType.SUBTLE_DIG:
            return random.choice([
                "Of course, that's one way to look at it...",
                "How... interesting.",
                "Well, some people think that, I suppose.",
            ])

        elif interaction_type == InteractionType.ASIDE:
            return f"*quietly to neighbor* Did you catch that?"

        else:  # STATEMENT or RESPONSE
            if random.random() < 0.4 and signature_phrases:
                return random.choice(signature_phrases)
            return dinner_behavior

    def _calculate_humor_level(self, content: str, persona: Dict) -> float:
        """Calculate humor level of content."""
        humor_words = ["funny", "hilarious", "joke", "laugh", "haha", "amusing"]
        base = 0.0
        for word in humor_words:
            if word in content.lower():
                base += 0.2

        # Persona humor tendency
        style = persona.get("speaking_style", "")
        if "witty" in style or "humorous" in style:
            base += 0.2

        return min(1.0, base)

    def _calculate_controversy_level(self, content: str, topic: ConversationTopic) -> float:
        """Calculate controversy level."""
        if topic == ConversationTopic.CONTROVERSIAL:
            return 0.7
        if topic == ConversationTopic.POLITICS:
            return 0.5

        controversy_words = ["disagree", "wrong", "ridiculous", "absurd"]
        base = 0.1
        for word in controversy_words:
            if word in content.lower():
                base += 0.15

        return min(1.0, base)

    def _calculate_intimacy_level(self, content: str, guest: GuestState) -> float:
        """Calculate intimacy/personal level."""
        intimate_words = ["family", "children", "personal", "remember when", "honestly"]
        base = 0.0
        for word in intimate_words:
            if word in content.lower():
                base += 0.15

        if guest.intoxication > 0.5:
            base += 0.2  # Tipsy people share more

        return min(1.0, base)

    async def _process_utterance_effects(self, utterance: Utterance):
        """Process effects of an utterance on other guests."""
        speaker = self.state.guests[utterance.speaker_id]

        for guest_id, guest in self.state.guests.items():
            if guest_id == utterance.speaker_id:
                continue

            # Update engagement
            if utterance.target_id == guest_id:
                guest.engagement = min(1.0, guest.engagement + 0.1)

            # Update relationships based on content type
            if utterance.interaction_type == InteractionType.COMPLIMENT and utterance.target_id == guest_id:
                old_comfort = guest.comfort_with.get(utterance.speaker_id, 0.5)
                guest.comfort_with[utterance.speaker_id] = min(1.0, old_comfort + 0.1)
                self.collector.record_relationship_change(
                    guest_id, utterance.speaker_id, "comfort", old_comfort, guest.comfort_with[utterance.speaker_id]
                )

            elif utterance.interaction_type == InteractionType.DISAGREEMENT and utterance.target_id == guest_id:
                old_tension = guest.tension_with.get(utterance.speaker_id, 0.0)
                guest.tension_with[utterance.speaker_id] = min(1.0, old_tension + 0.1)
                self.collector.record_relationship_change(
                    guest_id, utterance.speaker_id, "tension", old_tension, guest.tension_with[utterance.speaker_id]
                )

            # Mood shifts
            if utterance.humor_level > 0.5:
                if guest.mood != GuestMood.AMUSED:
                    old_mood = guest.mood
                    guest.mood = GuestMood.AMUSED
                    self.collector.record_mood_change(guest_id, old_mood, GuestMood.AMUSED, f"amused by {speaker.name}")
                    if self.on_mood_change:
                        self.on_mood_change(guest_id, old_mood, GuestMood.AMUSED)

            if utterance.controversy_level > 0.6:
                # Tension rising hook
                if self.state.overall_tension > 0.5:
                    context = self._build_hook_context(HookType.TENSION_RISING, None)
                    await self._trigger_hook(HookType.TENSION_RISING, context)

    async def _maybe_change_topic(self):
        """Maybe change the conversation topic."""
        # Pick a new topic not exhausted
        available = [t for t in ConversationTopic if t not in self.state.topics_exhausted]
        if not available:
            self.state.topics_exhausted.clear()
            available = list(ConversationTopic)

        old_topic = self.state.current_topic
        new_topic = random.choice(available)

        # Topic change hook
        context = self._build_hook_context(HookType.TOPIC_CHANGE, None)
        context.proposed_content = new_topic.value
        result = await self._trigger_hook(HookType.TOPIC_CHANGE, context)

        if result.new_topic:
            new_topic = result.new_topic

        self.state.current_topic = new_topic

        if self.on_topic_change:
            self.on_topic_change(old_topic, new_topic)

    async def _check_events(self):
        """Check for random dinner events."""
        if random.random() < 0.1:  # 10% chance per round
            event_types = [
                ("wine_refill", "The sommelier refills everyone's glasses"),
                ("dish_arrives", f"The {'appetizer' if self.state.course == 1 else 'main course' if self.state.course == 2 else 'dessert'} arrives"),
                ("phone_buzzes", f"{random.choice(list(self.state.guests.values())).name}'s phone buzzes"),
                ("laughter", "The table erupts in laughter"),
                ("clink", "Glasses clink around the table"),
            ]

            event_type, description = random.choice(event_types)
            event = DinnerEvent(
                id=f"evt_{len(self.state.events):03d}",
                event_type=event_type,
                description=description,
                affects=list(self.state.guests.keys()),
            )

            self.state.events.append(event)
            self.collector.record_event(event)

            if self.on_event:
                self.on_event(event)

            # Wine refills increase intoxication
            if event_type == "wine_refill":
                for guest in self.state.guests.values():
                    guest.intoxication = min(1.0, guest.intoxication + 0.1)
                    if guest.intoxication > 0.6 and guest.mood != GuestMood.TIPSY:
                        old_mood = guest.mood
                        guest.mood = GuestMood.TIPSY
                        self.collector.record_mood_change(guest.id, old_mood, GuestMood.TIPSY, "wine")

    def _update_party_dynamics(self):
        """Update overall party dynamics."""
        # Calculate overall energy
        energies = [g.energy for g in self.state.guests.values()]
        self.state.overall_energy = sum(energies) / len(energies) if energies else 0.5

        # Calculate overall tension
        all_tensions = []
        for guest in self.state.guests.values():
            all_tensions.extend(guest.tension_with.values())
        self.state.overall_tension = sum(all_tensions) / len(all_tensions) if all_tensions else 0.0

        # Energy decay
        for guest in self.state.guests.values():
            guest.energy = max(0.1, guest.energy - 0.02)

    def _build_hook_context(self, hook_type: HookType, speaker_id: Optional[str]) -> HookContext:
        """Build context for hooks."""
        recent = []
        for utt_id in self.state.conversation_flow[-5:]:
            if utt_id in self.state.all_utterances:
                recent.append(self.state.all_utterances[utt_id])

        next_idx = self.state.seating_order.index(speaker_id) + 1 if speaker_id and speaker_id in self.state.seating_order else 0
        next_speaker = self.state.seating_order[next_idx % len(self.state.seating_order)] if self.state.seating_order else None

        return HookContext(
            hook_type=hook_type,
            current_speaker=speaker_id,
            next_speaker=next_speaker,
            current_topic=self.state.current_topic,
            tension_level=self.state.overall_tension,
            recent_utterances=recent,
            guests=self.state.guests,
        )

    # === Human Intervention Methods ===

    def edit_last_utterance(self, new_content: str):
        """Edit the last utterance."""
        if self.state.conversation_flow:
            last_id = self.state.conversation_flow[-1]
            utt = self.state.all_utterances.get(last_id)
            if utt:
                utt.original_content = utt.content
                utt.content = new_content
                utt.edited = True

    def inject_event(self, event_type: str, description: str, affects: Optional[List[str]] = None):
        """Inject a custom event."""
        event = DinnerEvent(
            id=f"evt_injected_{len(self.state.events):03d}",
            event_type=event_type,
            description=description,
            triggered_by="human",
            affects=affects or list(self.state.guests.keys()),
        )
        self.state.events.append(event)
        self.collector.record_event(event)
        if self.on_event:
            self.on_event(event)

    def set_topic(self, topic: ConversationTopic):
        """Set the current topic."""
        old_topic = self.state.current_topic
        self.state.current_topic = topic
        if self.on_topic_change:
            self.on_topic_change(old_topic, topic)

    def set_guest_mood(self, guest_id: str, mood: GuestMood):
        """Set a guest's mood."""
        if guest_id in self.state.guests:
            guest = self.state.guests[guest_id]
            old_mood = guest.mood
            guest.mood = mood
            self.collector.record_mood_change(guest_id, old_mood, mood, "human_intervention")
            if self.on_mood_change:
                self.on_mood_change(guest_id, old_mood, mood)

    def add_guest_impression(self, observer_id: str, subject_id: str, impression: str):
        """Add an impression one guest has of another."""
        if observer_id in self.state.guests:
            self.state.guests[observer_id].impressions[subject_id] = impression

    # === Query Methods ===

    def get_transcript(self) -> List[Dict[str, Any]]:
        """Get the conversation transcript."""
        return [
            {
                "speaker": self.state.all_utterances[uid].speaker_name,
                "content": self.state.all_utterances[uid].content,
                "type": self.state.all_utterances[uid].interaction_type.value,
            }
            for uid in self.state.conversation_flow
            if uid in self.state.all_utterances
        ]

    def get_guest_summary(self, guest_id: str) -> Dict[str, Any]:
        """Get a summary of a guest's evening."""
        if guest_id not in self.state.guests:
            return {}

        guest = self.state.guests[guest_id]
        return {
            "name": guest.name,
            "mood": guest.mood.value,
            "energy": guest.energy,
            "intoxication": guest.intoxication,
            "utterances_count": len(guest.utterances_made),
            "impressions_formed": guest.impressions,
            "comfort_with": dict(guest.comfort_with),
            "tension_with": dict(guest.tension_with),
        }

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data."""
        return self.collector.get_summary()

    def export_data(self, path: str):
        """Export all collected data."""
        self.collector.export_json(path)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_dinner_party(
    guests: List[str],
    party_name: str = "An Evening Together",
    setting: str = "elegant_dining_room",
    host: Optional[str] = None,
) -> DinnerPartyEngine:
    """Create a dinner party simulation."""
    return DinnerPartyEngine(
        party_name=party_name,
        guest_ids=guests,
        setting=setting,
        host_id=host,
    )


def create_random_dinner_party(
    guest_count: int = 8,
    domains: Optional[List[PersonaDomain]] = None,
) -> DinnerPartyEngine:
    """Create a dinner party with random guests."""
    from .famous_personas import get_domain_mix, get_random_guests

    if domains:
        guests = get_domain_mix(domains, per_domain=guest_count // len(domains))
    else:
        guests = get_random_guests(guest_count)

    return create_dinner_party(guests)


def create_clash_dinner_party(theme: str = "tech_vs_arts") -> DinnerPartyEngine:
    """Create a dinner party with intentional personality clashes."""
    clash_configs = {
        "tech_vs_arts": {
            "guests": ["elon_musk", "taylor_swift", "mark_zuckerberg", "beyonce",
                      "sam_altman", "steven_spielberg", "jensen_huang", "rihanna"],
            "name": "Silicon Valley Meets Hollywood",
        },
        "old_vs_new": {
            "guests": ["warren_buffett", "elon_musk", "jane_goodall", "greta_thunberg",
                      "noam_chomsky", "joe_rogan", "angela_merkel", "volodymyr_zelenskyy"],
            "name": "Generations Collide",
        },
        "rivals": {
            "guests": ["elon_musk", "mark_zuckerberg", "sam_altman", "dario_amodei",
                      "yann_lecun", "yoshua_bengio", "jeff_bezos", "jensen_huang"],
            "name": "Tech Titans Dinner",
        },
        "intellectual_clash": {
            "guests": ["jordan_peterson", "slavoj_zizek", "noam_chomsky", "steven_pinker",
                      "neil_degrasse_tyson", "joe_rogan", "anderson_cooper", "trevor_noah"],
            "name": "Battle of Ideas",
        },
    }

    config = clash_configs.get(theme, clash_configs["tech_vs_arts"])
    return create_dinner_party(config["guests"], party_name=config["name"])
