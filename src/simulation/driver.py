"""
Simulation Driver

High-level API for driving multi-agent simulations:
- Spawn characters from templates
- Make characters perform actions
- Orchestrate multi-character scenarios
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable, Union

from .templates import PersonaTemplate, TemplateLoader, get_template_loader

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of actions a character can perform."""

    # Communication
    SAY = "say"                     # Say something (broadcast/multicast)
    TELL = "tell"                   # Tell a specific character something
    ASK = "ask"                     # Ask a question
    REPLY = "reply"                 # Reply to a message

    # Cognitive
    THINK = "think"                 # Internal reasoning/deliberation
    DECIDE = "decide"               # Make a decision
    PLAN = "plan"                   # Create a plan
    REFLECT = "reflect"             # Reflect on events

    # Social
    PROPOSE = "propose"             # Propose something to others
    VOTE = "vote"                   # Vote on a proposal
    AGREE = "agree"                 # Express agreement
    DISAGREE = "disagree"           # Express disagreement
    NEGOTIATE = "negotiate"         # Engage in negotiation

    # Task
    DELEGATE = "delegate"           # Delegate a task
    EXECUTE = "execute"             # Execute a task
    REPORT = "report"               # Report on task status

    # Emotional
    REACT = "react"                 # Emotional reaction
    EXPRESS = "express"             # Express emotion/state

    # Meta
    OBSERVE = "observe"             # Observe the environment
    REMEMBER = "remember"           # Recall from memory
    LEARN = "learn"                 # Learn new information
    UPDATE_BELIEF = "update_belief" # Update beliefs


@dataclass
class ActionResult:
    """Result of a character action."""
    success: bool
    action_type: ActionType
    character_id: str
    output: Any = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "action_type": self.action_type.value,
            "character_id": self.character_id,
            "output": self.output,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CharacterAction:
    """An action to be performed by a character."""
    action_type: ActionType
    content: Any
    target: Optional[str] = None  # Target character ID for directed actions
    context: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def say(cls, content: str, topic: Optional[str] = None) -> "CharacterAction":
        """Create a 'say' action."""
        return cls(ActionType.SAY, content, context={"topic": topic} if topic else {})

    @classmethod
    def tell(cls, target: str, content: str) -> "CharacterAction":
        """Create a 'tell' action directed at a specific character."""
        return cls(ActionType.TELL, content, target=target)

    @classmethod
    def ask(cls, target: str, question: str) -> "CharacterAction":
        """Create an 'ask' action."""
        return cls(ActionType.ASK, question, target=target)

    @classmethod
    def think(cls, topic: str) -> "CharacterAction":
        """Create a 'think' action for internal deliberation."""
        return cls(ActionType.THINK, topic)

    @classmethod
    def propose(cls, proposal: str, targets: Optional[List[str]] = None) -> "CharacterAction":
        """Create a 'propose' action."""
        return cls(ActionType.PROPOSE, proposal, context={"targets": targets or []})

    @classmethod
    def vote(cls, proposal_id: str, decision: str, rationale: str = "") -> "CharacterAction":
        """Create a 'vote' action."""
        return cls(ActionType.VOTE, {
            "proposal_id": proposal_id,
            "decision": decision,
            "rationale": rationale,
        })

    @classmethod
    def delegate(cls, target: str, task: Dict[str, Any]) -> "CharacterAction":
        """Create a 'delegate' action."""
        return cls(ActionType.DELEGATE, task, target=target)

    @classmethod
    def react(cls, emotion: str, to_event: str) -> "CharacterAction":
        """Create a 'react' action."""
        return cls(ActionType.REACT, {"emotion": emotion, "event": to_event})

    @classmethod
    def observe(cls, what: str = "environment") -> "CharacterAction":
        """Create an 'observe' action."""
        return cls(ActionType.OBSERVE, what)

    @classmethod
    def update_belief(cls, topic: str, belief: float, evidence: Optional[str] = None) -> "CharacterAction":
        """Create an 'update_belief' action."""
        return cls(ActionType.UPDATE_BELIEF, {
            "topic": topic,
            "belief": belief,
            "evidence": evidence,
        })


@dataclass
class Character:
    """A character in the simulation."""
    id: str
    name: str
    template: PersonaTemplate

    # State
    is_active: bool = True
    beliefs: Dict[str, float] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    pending_actions: List[CharacterAction] = field(default_factory=list)

    # History
    action_history: List[ActionResult] = field(default_factory=list)
    message_history: List[Dict[str, Any]] = field(default_factory=list)

    # Backing agent (if connected to messaging system)
    _agent: Optional[Any] = None

    def __post_init__(self):
        """Initialize beliefs from template."""
        if self.template.traits:
            self.beliefs.update(self.template.traits)

    async def perform(self, action: CharacterAction) -> ActionResult:
        """Perform an action."""
        try:
            result = await self._execute_action(action)
            self.action_history.append(result)
            return result
        except Exception as e:
            result = ActionResult(
                success=False,
                action_type=action.action_type,
                character_id=self.id,
                error=str(e),
            )
            self.action_history.append(result)
            return result

    async def _execute_action(self, action: CharacterAction) -> ActionResult:
        """Execute an action."""
        output = None

        if action.action_type == ActionType.SAY:
            output = await self._do_say(action)
        elif action.action_type == ActionType.TELL:
            output = await self._do_tell(action)
        elif action.action_type == ActionType.ASK:
            output = await self._do_ask(action)
        elif action.action_type == ActionType.THINK:
            output = await self._do_think(action)
        elif action.action_type == ActionType.PROPOSE:
            output = await self._do_propose(action)
        elif action.action_type == ActionType.VOTE:
            output = await self._do_vote(action)
        elif action.action_type == ActionType.DELEGATE:
            output = await self._do_delegate(action)
        elif action.action_type == ActionType.REACT:
            output = await self._do_react(action)
        elif action.action_type == ActionType.OBSERVE:
            output = await self._do_observe(action)
        elif action.action_type == ActionType.UPDATE_BELIEF:
            output = await self._do_update_belief(action)
        else:
            output = {"action": action.action_type.value, "content": action.content}

        # Record in memory
        self.memory.append({
            "type": "action",
            "action": action.action_type.value,
            "content": action.content,
            "output": output,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return ActionResult(
            success=True,
            action_type=action.action_type,
            character_id=self.id,
            output=output,
        )

    async def _do_say(self, action: CharacterAction) -> dict:
        """Execute a say action."""
        content = action.content
        topic = action.context.get("topic", "general")

        if self._agent:
            from ..messaging import MessageType
            await self._agent.publish(
                f"discussions:{topic}",
                {"speaker": self.name, "content": content},
            )

        return {"said": content, "topic": topic}

    async def _do_tell(self, action: CharacterAction) -> dict:
        """Execute a tell action."""
        if self._agent and action.target:
            from ..messaging import MessageType
            await self._agent.send(
                action.target,
                MessageType.REQUEST,
                {"from": self.name, "message": action.content},
            )

        return {"told": action.target, "content": action.content}

    async def _do_ask(self, action: CharacterAction) -> dict:
        """Execute an ask action."""
        if self._agent and action.target:
            from ..messaging import MessageType
            await self._agent.send(
                action.target,
                MessageType.REQUEST,
                {"action": "query", "query": action.content, "from": self.name},
            )

        return {"asked": action.target, "question": action.content}

    async def _do_think(self, action: CharacterAction) -> dict:
        """Execute a think action (internal deliberation)."""
        # This could be expanded to use LLM for actual reasoning
        thought = f"{self.name} is thinking about: {action.content}"
        return {"thought": thought, "topic": action.content}

    async def _do_propose(self, action: CharacterAction) -> dict:
        """Execute a propose action."""
        proposal_id = f"proposal-{uuid.uuid4().hex[:8]}"
        targets = action.context.get("targets", [])

        if self._agent:
            from ..messaging import MessageType
            for target_id in targets:
                await self._agent.send(
                    target_id,
                    MessageType.PROPOSE,
                    {
                        "id": proposal_id,
                        "content": action.content,
                        "from": self.name,
                    },
                )

        return {"proposal_id": proposal_id, "content": action.content, "sent_to": targets}

    async def _do_vote(self, action: CharacterAction) -> dict:
        """Execute a vote action."""
        vote_data = action.content
        return {"voted": vote_data}

    async def _do_delegate(self, action: CharacterAction) -> dict:
        """Execute a delegate action."""
        if self._agent and action.target:
            from ..messaging import MessageType
            await self._agent.send(
                action.target,
                MessageType.DELEGATE,
                {"task": action.content, "from": self.name},
            )

        return {"delegated_to": action.target, "task": action.content}

    async def _do_react(self, action: CharacterAction) -> dict:
        """Execute a react action."""
        return {"reaction": action.content}

    async def _do_observe(self, action: CharacterAction) -> dict:
        """Execute an observe action."""
        return {"observed": action.content, "observer": self.name}

    async def _do_update_belief(self, action: CharacterAction) -> dict:
        """Execute an update_belief action."""
        data = action.content
        topic = data.get("topic")
        belief = data.get("belief")

        if topic and belief is not None:
            self.beliefs[topic] = max(0.0, min(1.0, float(belief)))

        return {"topic": topic, "new_belief": self.beliefs.get(topic)}


class SimulationDriver:
    """
    High-level driver for multi-agent simulations.

    Provides APIs for:
    - Loading persona templates
    - Spawning characters from templates
    - Making individual characters perform actions
    - Running multi-character scenarios
    """

    def __init__(self, broker: Optional[Any] = None):
        """Initialize the simulation driver.

        Args:
            broker: Optional MessageBroker for connected messaging
        """
        self.broker = broker
        self.template_loader = get_template_loader()
        self.characters: Dict[str, Character] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}

    # -------------------------------------------------------------------------
    # Template Loading
    # -------------------------------------------------------------------------

    def load_templates(self) -> Dict[str, PersonaTemplate]:
        """Load all available templates."""
        self.template_loader.load_all()
        return {t.id: t for t in self.template_loader.list_all()}

    def get_template(self, template_id: str) -> Optional[PersonaTemplate]:
        """Get a template by ID."""
        return self.template_loader.get(template_id)

    def list_templates(self) -> List[str]:
        """List all template IDs."""
        return self.template_loader.list_ids()

    def search_templates(self, query: str) -> List[PersonaTemplate]:
        """Search templates by name."""
        return self.template_loader.search(query)

    # -------------------------------------------------------------------------
    # Character Management
    # -------------------------------------------------------------------------

    def spawn(
        self,
        template_id: str,
        character_id: Optional[str] = None,
    ) -> Character:
        """Spawn a character from a template.

        Args:
            template_id: ID of the template to use
            character_id: Optional custom ID for the character

        Returns:
            The spawned Character
        """
        template = self.template_loader.get(template_id)
        if not template:
            raise ValueError(f"Template not found: {template_id}")

        char_id = character_id or f"{template_id}-{uuid.uuid4().hex[:6]}"

        character = Character(
            id=char_id,
            name=template.name,
            template=template,
        )

        self.characters[char_id] = character
        logger.info(f"Spawned character: {char_id} ({template.name})")

        return character

    def spawn_from_templates(
        self,
        template_ids: List[str],
    ) -> List[Character]:
        """Spawn multiple characters from templates."""
        return [self.spawn(tid) for tid in template_ids]

    def spawn_all(self) -> List[Character]:
        """Spawn all available templates as characters."""
        return self.spawn_from_templates(self.list_templates())

    def get_character(self, character_id: str) -> Optional[Character]:
        """Get a character by ID."""
        return self.characters.get(character_id)

    def get_characters_by_category(self, category: str) -> List[Character]:
        """Get characters by their template category."""
        return [
            c for c in self.characters.values()
            if c.template.category.value == category
        ]

    def list_characters(self) -> List[str]:
        """List all character IDs."""
        return list(self.characters.keys())

    def remove_character(self, character_id: str) -> bool:
        """Remove a character from the simulation."""
        if character_id in self.characters:
            del self.characters[character_id]
            return True
        return False

    # -------------------------------------------------------------------------
    # Character Actions
    # -------------------------------------------------------------------------

    async def do(
        self,
        character_id: str,
        action: Union[CharacterAction, str],
        **kwargs,
    ) -> ActionResult:
        """Make a character perform an action.

        Args:
            character_id: ID of the character
            action: Either a CharacterAction or an action type string
            **kwargs: Additional arguments for the action

        Returns:
            ActionResult with the outcome
        """
        character = self.characters.get(character_id)
        if not character:
            return ActionResult(
                success=False,
                action_type=ActionType.SAY,
                character_id=character_id,
                error=f"Character not found: {character_id}",
            )

        # Convert string to action if needed
        if isinstance(action, str):
            action = self._parse_action(action, **kwargs)

        return await character.perform(action)

    def _parse_action(self, action_str: str, **kwargs) -> CharacterAction:
        """Parse an action string into a CharacterAction."""
        try:
            action_type = ActionType(action_str.lower())
        except ValueError:
            action_type = ActionType.SAY

        content = kwargs.get("content", kwargs.get("message", ""))
        target = kwargs.get("target", kwargs.get("to"))

        return CharacterAction(
            action_type=action_type,
            content=content,
            target=target,
            context=kwargs,
        )

    # Convenience methods for common actions
    async def say(self, character_id: str, message: str, topic: str = "general") -> ActionResult:
        """Make a character say something."""
        return await self.do(character_id, CharacterAction.say(message, topic))

    async def tell(self, from_id: str, to_id: str, message: str) -> ActionResult:
        """Make a character tell something to another character."""
        return await self.do(from_id, CharacterAction.tell(to_id, message))

    async def ask(self, from_id: str, to_id: str, question: str) -> ActionResult:
        """Make a character ask another character a question."""
        return await self.do(from_id, CharacterAction.ask(to_id, question))

    async def think(self, character_id: str, topic: str) -> ActionResult:
        """Make a character think about something."""
        return await self.do(character_id, CharacterAction.think(topic))

    async def propose(
        self,
        character_id: str,
        proposal: str,
        targets: Optional[List[str]] = None
    ) -> ActionResult:
        """Make a character propose something."""
        return await self.do(character_id, CharacterAction.propose(proposal, targets))

    async def delegate(
        self,
        from_id: str,
        to_id: str,
        task: Dict[str, Any]
    ) -> ActionResult:
        """Make a character delegate a task."""
        return await self.do(from_id, CharacterAction.delegate(to_id, task))

    async def update_belief(
        self,
        character_id: str,
        topic: str,
        belief: float,
        evidence: Optional[str] = None
    ) -> ActionResult:
        """Update a character's belief."""
        return await self.do(
            character_id,
            CharacterAction.update_belief(topic, belief, evidence)
        )

    # -------------------------------------------------------------------------
    # Batch Actions
    # -------------------------------------------------------------------------

    async def all_say(self, message: str, topic: str = "general") -> List[ActionResult]:
        """Make all characters say something."""
        tasks = [self.say(cid, message, topic) for cid in self.characters]
        return await asyncio.gather(*tasks)

    async def all_think(self, topic: str) -> List[ActionResult]:
        """Make all characters think about something."""
        tasks = [self.think(cid, topic) for cid in self.characters]
        return await asyncio.gather(*tasks)

    async def category_say(
        self,
        category: str,
        message: str,
        topic: str = "general"
    ) -> List[ActionResult]:
        """Make all characters in a category say something."""
        chars = self.get_characters_by_category(category)
        tasks = [self.say(c.id, message, topic) for c in chars]
        return await asyncio.gather(*tasks)

    # -------------------------------------------------------------------------
    # Scenarios
    # -------------------------------------------------------------------------

    async def run_dialogue(
        self,
        participants: List[str],
        topic: str,
        rounds: int = 3,
    ) -> List[ActionResult]:
        """Run a multi-turn dialogue between characters.

        Args:
            participants: List of character IDs
            topic: Topic of discussion
            rounds: Number of dialogue rounds

        Returns:
            List of action results from the dialogue
        """
        results = []

        for round_num in range(rounds):
            for participant_id in participants:
                character = self.characters.get(participant_id)
                if not character:
                    continue

                # Each character thinks then speaks
                think_result = await self.think(participant_id, topic)
                results.append(think_result)

                say_result = await self.say(
                    participant_id,
                    f"[Round {round_num + 1}] My perspective on {topic}",
                    topic
                )
                results.append(say_result)

        return results

    async def run_debate(
        self,
        pro_characters: List[str],
        con_characters: List[str],
        topic: str,
        rounds: int = 2,
    ) -> List[ActionResult]:
        """Run a debate between two groups of characters.

        Args:
            pro_characters: Characters arguing for the position
            con_characters: Characters arguing against
            topic: Debate topic
            rounds: Number of rounds

        Returns:
            List of action results from the debate
        """
        results = []

        for round_num in range(rounds):
            # Pro side speaks
            for char_id in pro_characters:
                result = await self.say(
                    char_id,
                    f"[PRO Round {round_num + 1}] Arguments supporting {topic}",
                    f"debate:{topic}"
                )
                results.append(result)

            # Con side responds
            for char_id in con_characters:
                result = await self.say(
                    char_id,
                    f"[CON Round {round_num + 1}] Arguments against {topic}",
                    f"debate:{topic}"
                )
                results.append(result)

        return results

    async def run_vote(
        self,
        proposal: str,
        proposer_id: str,
        voter_ids: List[str],
    ) -> Dict[str, Any]:
        """Run a voting scenario.

        Args:
            proposal: The proposal text
            proposer_id: Character making the proposal
            voter_ids: Characters who will vote

        Returns:
            Voting results with tally
        """
        # Proposer makes proposal
        propose_result = await self.propose(proposer_id, proposal, voter_ids)
        proposal_id = propose_result.output.get("proposal_id", "unknown")

        # Collect votes
        votes = {}
        for voter_id in voter_ids:
            character = self.characters.get(voter_id)
            if not character:
                continue

            # Simple vote based on random or belief
            decision = "approve"  # Could be more sophisticated
            vote_result = await self.do(
                voter_id,
                CharacterAction.vote(proposal_id, decision, "Based on my assessment")
            )
            votes[voter_id] = vote_result.output

        # Tally
        approve_count = sum(1 for v in votes.values() if v.get("voted", {}).get("decision") == "approve")
        reject_count = len(votes) - approve_count

        return {
            "proposal_id": proposal_id,
            "proposal": proposal,
            "proposer": proposer_id,
            "votes": votes,
            "approved": approve_count,
            "rejected": reject_count,
            "passed": approve_count > reject_count,
        }

    # -------------------------------------------------------------------------
    # State & Inspection
    # -------------------------------------------------------------------------

    def get_character_state(self, character_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a character."""
        character = self.characters.get(character_id)
        if not character:
            return None

        return {
            "id": character.id,
            "name": character.name,
            "is_active": character.is_active,
            "beliefs": character.beliefs.copy(),
            "memory_size": len(character.memory),
            "action_count": len(character.action_history),
            "template": {
                "id": character.template.id,
                "category": character.template.category.value,
            },
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all characters."""
        return {
            cid: self.get_character_state(cid)
            for cid in self.characters
        }

    def get_action_history(
        self,
        character_id: Optional[str] = None,
        action_type: Optional[ActionType] = None,
        limit: int = 50,
    ) -> List[ActionResult]:
        """Get action history, optionally filtered."""
        if character_id:
            character = self.characters.get(character_id)
            if not character:
                return []
            history = character.action_history
        else:
            # Combine all histories
            history = []
            for c in self.characters.values():
                history.extend(c.action_history)
            history.sort(key=lambda r: r.timestamp, reverse=True)

        if action_type:
            history = [r for r in history if r.action_type == action_type]

        return history[:limit]

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def on_action(
        self,
        action_type: ActionType,
        handler: Callable[[ActionResult], Awaitable[None]],
    ):
        """Register a handler for action events."""
        if action_type.value not in self._event_handlers:
            self._event_handlers[action_type.value] = []
        self._event_handlers[action_type.value].append(handler)

    async def _emit_action(self, result: ActionResult):
        """Emit an action event to handlers."""
        handlers = self._event_handlers.get(result.action_type.value, [])
        for handler in handlers:
            try:
                await handler(result)
            except Exception as e:
                logger.error(f"Handler error: {e}")

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def clear(self):
        """Clear all characters."""
        self.characters.clear()

    def reset(self):
        """Reset the simulation state."""
        for character in self.characters.values():
            character.memory.clear()
            character.action_history.clear()
            character.message_history.clear()
            character.beliefs = character.template.traits.copy() if character.template.traits else {}
