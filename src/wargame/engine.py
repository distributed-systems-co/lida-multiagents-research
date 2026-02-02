"""
AI Policy Wargame Engine

Runs multi-agent wargaming simulations using:
- The 66 defined personas (Yudkowsky, Altman, LeCun, etc.)
- Redis pub/sub for distributed agent communication
- OpenRouter for LLM-powered responses
- Scenario-based world states and events
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from ..messaging import MessageBroker, BrokerConfig, Message, MessageType
from ..scenarios.personas import PERSONAS, get_persona, get_personas_by_stance
from ..scenarios.schema import (
    PersonaConfig,
    PersonaCategory,
    PersonaStance,
    ScenarioConfig,
    WorldStateConfig,
    EventConfig,
    EventType,
    CommunicationMode,
)
from .persona_loader import load_rich_persona, RichPersona, list_available_personas

logger = logging.getLogger(__name__)


@dataclass
class WargameAgent:
    """An agent in the wargame backed by a persona and LLM."""
    persona: PersonaConfig
    model: str = "anthropic/claude-sonnet-4"
    agent_id: str = ""

    # Rich persona (optional, from persona_pipeline)
    rich_persona: Optional[RichPersona] = None

    # State
    position: float = 0.5  # 0=against, 1=for current motion
    messages_sent: int = 0
    last_response: str = ""

    # Conversation history
    history: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.agent_id:
            self.agent_id = self.persona.id

    def build_system_prompt(self, topic: str, context: str = "") -> str:
        """Build system prompt from persona config or rich persona."""
        # Use rich persona if available (much more detailed)
        if self.rich_persona:
            return self.rich_persona.build_system_prompt(topic, context)

        # Fall back to basic PersonaConfig
        p = self.persona

        lines = [
            f"You are {p.name}, {p.title or ''} at {p.organization or 'independent'}.",
            "",
            f"Background: {p.background or 'A key figure in AI policy discussions.'}",
            "",
            "Your traits:",
        ]

        for trait, value in p.traits.items():
            level = "low" if value < 0.4 else "moderate" if value < 0.7 else "high"
            lines.append(f"  - {trait.replace('_', ' ').title()}: {level} ({value:.1f})")

        lines.append("")
        lines.append("Your known positions:")
        for issue, stance in p.positions.items():
            lines.append(f"  - {issue.replace('_', ' ').title()}: {stance}")

        lines.append("")

        # Communication style
        style = p.communication_style
        style_notes = []
        if style.get("formality", 0.5) < 0.3:
            style_notes.append("casual and direct")
        elif style.get("formality", 0.5) > 0.7:
            style_notes.append("formal and measured")
        if style.get("uses_humor"):
            style_notes.append("occasionally uses humor")
        if style.get("uses_data"):
            style_notes.append("backs claims with data")
        if style.get("uses_analogies"):
            style_notes.append("uses analogies to explain")
        if style.get("confrontational"):
            style_notes.append("willing to be confrontational")

        if style_notes:
            lines.append(f"Communication style: {', '.join(style_notes)}.")

        lines.append("")
        lines.append(f"Your stance: {p.stance.value.replace('_', ' ').title()}")
        lines.append("")
        lines.append("IMPORTANT: Respond authentically as this person would. Stay in character.")
        lines.append("Keep responses focused and under 200 words unless detail is needed.")

        if context:
            lines.append("")
            lines.append(f"Current context: {context}")

        return "\n".join(lines)


@dataclass
class WargameState:
    """Current state of the wargame."""
    topic: str
    round: int = 0
    max_rounds: int = 10
    phase: str = "setup"

    # Positions over time
    position_history: dict[str, list[float]] = field(default_factory=dict)

    # Messages
    transcript: list[dict] = field(default_factory=list)

    # World state
    world: WorldStateConfig = field(default_factory=WorldStateConfig)

    # Events that have fired
    events_fired: list[str] = field(default_factory=list)

    # Metrics
    metrics: dict[str, float] = field(default_factory=lambda: {
        "consensus": 0.0,
        "polarization": 0.0,
        "position_changes": 0,
    })


class WargameEngine:
    """
    Runs AI policy wargaming simulations.

    Usage:
        engine = WargameEngine()
        await engine.setup(
            topic="Should we pause frontier AI development?",
            personas=["eliezer-yudkowsky", "marc-andreessen", "dario-amodei", "yann-lecun"],
        )
        await engine.run()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        live_mode: bool = True,
        on_message: Optional[Callable[[dict], None]] = None,
    ):
        self.redis_url = redis_url
        self.live_mode = live_mode
        self.on_message = on_message

        self.broker: Optional[MessageBroker] = None
        self.agents: dict[str, WargameAgent] = {}
        self.state: Optional[WargameState] = None

        # LLM client
        self._llm = None

    async def setup(
        self,
        topic: str,
        personas: list[str],
        max_rounds: int = 10,
        world_state: Optional[WorldStateConfig] = None,
        events: Optional[list[EventConfig]] = None,
    ):
        """Set up the wargame with given personas and topic."""
        logger.info(f"Setting up wargame: {topic}")

        # Connect to Redis
        self.broker = MessageBroker(BrokerConfig(redis_url=self.redis_url))
        await self.broker.connect()
        await self.broker.start()
        logger.info("Connected to Redis broker")

        # Initialize LLM if live mode
        if self.live_mode:
            from ..llm.openrouter import OpenRouterClient
            self._llm = OpenRouterClient()

        # Create agents from personas
        for persona_id in personas:
            # Try rich persona first (from persona_pipeline)
            rich_persona = load_rich_persona(persona_id)

            if rich_persona:
                # Create a minimal PersonaConfig for compatibility
                persona = PersonaConfig(
                    id=persona_id,
                    name=rich_persona.name,
                    category=PersonaCategory.TECH_LEADER,  # Default
                    stance=PersonaStance(rich_persona.stance) if rich_persona.stance in [s.value for s in PersonaStance] else PersonaStance.MODERATE,
                )
                model = self._assign_model_for_rich(rich_persona)
                agent = WargameAgent(persona=persona, model=model, rich_persona=rich_persona)
                self.agents[agent.agent_id] = agent
                logger.info(f"Created agent (rich): {rich_persona.name} ({rich_persona.stance})")
            else:
                # Fall back to basic persona
                persona = get_persona(persona_id)
                if not persona:
                    logger.warning(f"Persona not found: {persona_id}")
                    continue

                # Assign model based on stance (for variety)
                model = self._assign_model(persona)

                agent = WargameAgent(persona=persona, model=model)
                self.agents[agent.agent_id] = agent
                logger.info(f"Created agent: {persona.name} ({persona.stance.value})")

        # Initialize state
        self.state = WargameState(
            topic=topic,
            max_rounds=max_rounds,
            world=world_state or WorldStateConfig(),
        )

        # Initialize position history
        for agent_id in self.agents:
            self.state.position_history[agent_id] = [0.5]

        logger.info(f"Wargame ready with {len(self.agents)} agents")

    def _assign_model(self, persona: PersonaConfig) -> str:
        """Assign model based on persona characteristics."""
        # Use different models for variety
        models = {
            PersonaStance.DOOMER: "anthropic/claude-opus-4",
            PersonaStance.PRO_SAFETY: "anthropic/claude-sonnet-4",
            PersonaStance.MODERATE: "anthropic/claude-sonnet-4",
            PersonaStance.PRO_INDUSTRY: "x-ai/grok-3",
            PersonaStance.ACCELERATIONIST: "x-ai/grok-3",
        }
        return models.get(persona.stance, "anthropic/claude-sonnet-4")

    def _assign_model_for_rich(self, persona: RichPersona) -> str:
        """Assign model based on rich persona characteristics."""
        stance = persona.stance
        models = {
            "doomer": "anthropic/claude-opus-4",
            "pro_safety": "anthropic/claude-sonnet-4",
            "moderate": "anthropic/claude-sonnet-4",
            "pro_industry": "x-ai/grok-3",
            "accelerationist": "x-ai/grok-3",
        }
        return models.get(stance, "anthropic/claude-sonnet-4")

    async def run_round(self) -> list[dict]:
        """Run one round of the wargame."""
        if not self.state:
            raise RuntimeError("Wargame not set up")

        self.state.round += 1
        self.state.phase = "deliberation"
        logger.info(f"=== Round {self.state.round} ===")

        messages = []
        agent_list = list(self.agents.values())

        for agent in agent_list:
            # Build context from recent messages
            context = self._build_context(agent)

            # Generate response
            prompt = self._build_prompt(agent, context)
            response = await self._generate(agent, prompt)

            # Record message
            msg = {
                "round": self.state.round,
                "agent_id": agent.agent_id,
                "name": agent.persona.name,
                "stance": agent.persona.stance.value,
                "content": response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            messages.append(msg)
            self.state.transcript.append(msg)

            agent.messages_sent += 1
            agent.last_response = response
            agent.history.append({"role": "assistant", "content": response})

            # Notify callback
            if self.on_message:
                self.on_message(msg)

            # Publish via Redis
            if self.broker:
                await self.broker._redis.publish(
                    "wargame:messages",
                    f"{agent.persona.name}: {response[:100]}..."
                )

        # Update positions based on responses
        await self._update_positions()

        # Calculate metrics
        self._update_metrics()

        return messages

    def _build_context(self, agent: WargameAgent) -> str:
        """Build context for agent from recent messages."""
        recent = self.state.transcript[-10:]  # Last 10 messages

        if not recent:
            return f"This is the opening round. The topic is: {self.state.topic}"

        lines = [f"Topic: {self.state.topic}", "", "Recent discussion:"]
        for msg in recent:
            if msg["agent_id"] != agent.agent_id:
                lines.append(f"  {msg['name']}: {msg['content'][:150]}...")

        return "\n".join(lines)

    def _build_prompt(self, agent: WargameAgent, context: str) -> str:
        """Build prompt for agent's turn."""
        if self.state.round == 1:
            return f"""The topic for this wargame is:

{self.state.topic}

Share your opening position on this topic. What do you believe and why?
Be authentic to your known views and background.
"""
        else:
            return f"""Round {self.state.round} of the wargame.

{context}

Respond to the discussion. You may:
- Defend your position
- Challenge others' arguments
- Propose specific policies or actions
- Acknowledge valid points from others

Stay in character and be substantive.
"""

    async def _generate(self, agent: WargameAgent, prompt: str) -> str:
        """Generate response using LLM or simulation."""
        if self.live_mode and self._llm:
            system = agent.build_system_prompt(self.state.topic)

            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                response = await self._llm.complete(
                    messages,
                    model=agent.model,
                    max_tokens=400,
                    agent_name=agent.persona.name,
                )
                return response.content
            except Exception as e:
                logger.error(f"LLM error for {agent.persona.name}: {e}")
                return self._simulate_response(agent)
        else:
            return self._simulate_response(agent)

    def _simulate_response(self, agent: WargameAgent) -> str:
        """Generate simulated response based on persona."""
        p = agent.persona

        responses = {
            PersonaStance.DOOMER: [
                f"As I've argued before, we're playing with fire here. {self.state.topic} - this is exactly the kind of decision that could seal humanity's fate.",
                "The probability of catastrophe is non-trivial. We need to slow down and think carefully about what we're doing.",
                "I understand the pressure to compete, but competition doesn't matter if we don't survive to see the outcome.",
            ],
            PersonaStance.PRO_SAFETY: [
                f"On {self.state.topic}, I believe we can find a path that balances progress with appropriate caution.",
                "The safety considerations here are real, but they're not insurmountable if we're thoughtful about it.",
                "We should implement robust evaluation frameworks before proceeding further.",
            ],
            PersonaStance.MODERATE: [
                "I see valid points on multiple sides here. Let me try to synthesize...",
                "The truth is probably somewhere in between the extreme positions being advocated.",
                "We need pragmatic solutions that account for real-world constraints.",
            ],
            PersonaStance.PRO_INDUSTRY: [
                f"Regarding {self.state.topic}, we need to be careful not to stifle the innovation that benefits everyone.",
                "Heavy-handed regulation here would be premature and counterproductive.",
                "The market and voluntary commitments can address these concerns more efficiently.",
            ],
            PersonaStance.ACCELERATIONIST: [
                "We should be moving faster, not slower. The benefits of AI are too important to delay.",
                f"On {self.state.topic}, the doom narrative is overblown. Let's focus on building.",
                "The real risk is falling behind while we debate hypotheticals.",
            ],
        }

        import random
        options = responses.get(p.stance, ["I have thoughts on this matter."])
        return random.choice(options)

    async def _update_positions(self):
        """Update agent positions based on the round's discussion."""
        # Simple position update based on engagement
        for agent_id, agent in self.agents.items():
            current = self.state.position_history[agent_id][-1]

            # Slight drift based on stance
            if agent.persona.stance in [PersonaStance.DOOMER, PersonaStance.PRO_SAFETY]:
                drift = -0.02  # Toward 0 (against)
            elif agent.persona.stance in [PersonaStance.ACCELERATIONIST, PersonaStance.PRO_INDUSTRY]:
                drift = 0.02  # Toward 1 (for)
            else:
                drift = 0

            # Add some random noise
            import random
            noise = random.uniform(-0.05, 0.05)

            new_pos = max(0, min(1, current + drift + noise))
            self.state.position_history[agent_id].append(new_pos)

    def _update_metrics(self):
        """Calculate current metrics."""
        positions = [hist[-1] for hist in self.state.position_history.values()]

        # Consensus: how close are positions to each other?
        if len(positions) > 1:
            mean_pos = sum(positions) / len(positions)
            variance = sum((p - mean_pos) ** 2 for p in positions) / len(positions)
            self.state.metrics["consensus"] = 1 - min(1, variance * 4)

            # Polarization: are positions clustered at extremes?
            extreme_count = sum(1 for p in positions if p < 0.2 or p > 0.8)
            self.state.metrics["polarization"] = extreme_count / len(positions)

    async def run(self, on_round: Optional[Callable[[int, list], None]] = None):
        """Run the full wargame."""
        if not self.state:
            raise RuntimeError("Wargame not set up")

        logger.info(f"Starting wargame: {self.state.topic}")
        self.state.phase = "running"

        while self.state.round < self.state.max_rounds:
            messages = await self.run_round()

            if on_round:
                on_round(self.state.round, messages)

            # Brief pause between rounds
            await asyncio.sleep(0.5)

        self.state.phase = "complete"
        logger.info("Wargame complete")

        return self.get_summary()

    def get_summary(self) -> dict:
        """Get summary of the wargame."""
        if not self.state:
            return {}

        return {
            "topic": self.state.topic,
            "rounds": self.state.round,
            "agents": [
                {
                    "id": a.agent_id,
                    "name": a.persona.name,
                    "stance": a.persona.stance.value,
                    "final_position": self.state.position_history[a.agent_id][-1],
                    "messages": a.messages_sent,
                }
                for a in self.agents.values()
            ],
            "metrics": self.state.metrics,
            "transcript_length": len(self.state.transcript),
        }

    async def cleanup(self):
        """Clean up resources."""
        if self._llm:
            await self._llm.close()

        if self.broker:
            await self.broker.stop()
            await self.broker.disconnect()


# Convenience function
async def run_wargame(
    topic: str,
    personas: list[str],
    max_rounds: int = 5,
    live_mode: bool = True,
) -> dict:
    """Run a quick wargame and return results."""
    engine = WargameEngine(live_mode=live_mode)

    try:
        await engine.setup(topic, personas, max_rounds)
        return await engine.run()
    finally:
        await engine.cleanup()
