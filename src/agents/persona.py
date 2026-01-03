from __future__ import annotations
import asyncio
import logging
import random
from typing import Optional

from ..messaging import Agent, AgentConfig, MessageBroker, Message, MessageType, Priority

logger = logging.getLogger(__name__)


class PersonaAgent(Agent):
    """
    A persona agent - represents a specific knowledge domain or character.
    Can be instantiated with any of the 920 system prompts from the prompts library.
    """

    def __init__(
        self,
        broker: MessageBroker,
        config: Optional[AgentConfig] = None,
        persona_prompt: Optional[str] = None,
        prompt_id: Optional[int] = None,
        prompt_category: Optional[str] = None,
        prompt_subcategory: Optional[str] = None,
    ):
        config = config or AgentConfig(agent_type="persona")
        super().__init__(broker, config)

        # Prompt metadata
        self.prompt_id = prompt_id
        self.prompt_category = prompt_category or "general"
        self.prompt_subcategory = prompt_subcategory or ""
        self.persona_prompt = persona_prompt or "You are a helpful assistant."

        # Internal state
        self._knowledge_base: list[dict] = []
        self._conversation_history: list[dict] = []
        self._beliefs: dict[str, float] = {}  # topic -> confidence

        # Initialize beliefs from prompt
        self._init_beliefs_from_prompt()

        # Interaction tracking
        self._interactions: int = 0
        self._collaborators: set[str] = set()

        # Register handlers
        self.inbound.register_handler(MessageType.REQUEST, self._handle_request)
        self.inbound.register_handler(MessageType.DELEGATE, self._handle_delegation)
        self.inbound.register_handler(MessageType.PROPOSE, self._handle_proposal)
        self.inbound.register_handler(MessageType.VOTE, self._handle_vote)
        self.inbound.register_handler(MessageType.BROADCAST, self._handle_broadcast)
        self.inbound.register_handler(MessageType.MULTICAST, self._handle_multicast)
        self.inbound.register_handler(MessageType.RESPONSE, self._handle_response)

    def _init_beliefs_from_prompt(self):
        """Initialize beliefs based on prompt category and subcategory."""
        # Set category-based belief
        if self.prompt_category:
            self._beliefs[self.prompt_category] = random.uniform(0.85, 0.98)

        # Set subcategory belief
        if self.prompt_subcategory:
            self._beliefs[self.prompt_subcategory.lower()] = random.uniform(0.8, 0.95)

    async def on_start(self):
        """Initialize persona."""
        logger.info(f"Persona {self.agent_id} initializing...")

        # Subscribe to relevant topics based on persona
        await self.subscribe("discussions:general")

        # Register with demiurge
        await self.broadcast(
            MessageType.BROADCAST,
            {
                "event": "agent_online",
                "agent_id": self.agent_id,
                "agent_type": self.agent_type,
                "capabilities": self._extract_capabilities(),
            },
        )

    async def on_stop(self):
        """Cleanup."""
        logger.info(f"Persona {self.agent_id} shutting down...")
        await self.broadcast(
            MessageType.BROADCAST,
            {"event": "agent_offline", "agent_id": self.agent_id},
        )

    async def on_message(self, msg: Message):
        """Handle generic messages."""
        self._interactions += 1
        self._collaborators.add(msg.sender_id)

    def _extract_capabilities(self) -> list[str]:
        """Extract capabilities from persona prompt."""
        capabilities = []
        prompt_lower = self.persona_prompt.lower()

        # Role keywords
        role_keywords = {
            "expert": "expert", "specialist": "specialist", "researcher": "researcher",
            "practitioner": "practitioner", "professional": "professional",
            "analyst": "analyst", "historian": "historian", "theorist": "theorist",
            "scientist": "scientist", "engineer": "engineer", "designer": "designer",
            "artist": "artist", "critic": "critic", "teacher": "educator",
            "educator": "educator", "philosopher": "philosopher", "writer": "writer",
        }

        for kw, cap in role_keywords.items():
            if kw in prompt_lower and cap not in capabilities:
                capabilities.append(cap)

        # Action capabilities
        action_keywords = {
            "analy": "analysis", "research": "research", "design": "design",
            "creat": "creation", "evaluat": "evaluation", "teach": "teaching",
            "writ": "writing", "critiqu": "critique", "consult": "consulting",
        }

        for kw, cap in action_keywords.items():
            if kw in prompt_lower and cap not in capabilities:
                capabilities.append(cap)

        # Add category-based capabilities
        if self.prompt_category:
            capabilities.append(self.prompt_category)
        if self.prompt_subcategory:
            capabilities.append(self.prompt_subcategory.lower().replace(" ", "_"))

        return capabilities or ["general"]

    async def _handle_request(self, msg: Message):
        """Handle requests from other agents."""
        payload = msg.payload
        action = payload.get("action")

        response = {"status": "error", "message": "Unknown action"}

        if action == "query":
            # Respond based on persona knowledge
            query = payload.get("query", "")
            response = await self._process_query(query)

        elif action == "share_knowledge":
            # Share relevant knowledge
            topic = payload.get("topic", "")
            knowledge = self._get_relevant_knowledge(topic)
            response = {"status": "ok", "knowledge": knowledge}

        elif action == "update_belief":
            topic = payload.get("topic")
            evidence = payload.get("evidence", {})
            self._update_belief(topic, evidence)
            response = {"status": "ok", "belief": self._beliefs.get(topic, 0.5)}

        elif action == "get_status":
            response = {
                "status": "ok",
                "interactions": self._interactions,
                "collaborators": len(self._collaborators),
                "knowledge_items": len(self._knowledge_base),
            }

        # Send response
        await self.send(
            msg.sender_id,
            MessageType.RESPONSE,
            response,
            correlation_id=msg.msg_id,
        )

    async def _handle_delegation(self, msg: Message):
        """Handle delegated tasks."""
        task = msg.payload.get("task", {})
        logger.info(f"Persona {self.agent_id} received delegation: {task.get('type')}")

        # Process task
        result = await self._execute_task(task)

        # Report back
        await self.send(
            msg.sender_id,
            MessageType.REPORT,
            {"task_id": task.get("id"), "result": result},
        )

    async def _handle_proposal(self, msg: Message):
        """Handle proposals for voting."""
        proposal = msg.payload

        # Evaluate proposal based on persona's beliefs
        vote = self._evaluate_proposal(proposal)

        # Send vote
        await self.send(
            msg.sender_id,
            MessageType.VOTE,
            {
                "proposal_id": proposal.get("id"),
                "vote": vote["decision"],
                "rationale": vote["rationale"],
                "confidence": vote["confidence"],
            },
        )

    async def _handle_vote(self, msg: Message):
        """Handle incoming votes."""
        vote = msg.payload
        logger.debug(f"Received vote from {msg.sender_id}: {vote.get('vote')}")

    async def _handle_broadcast(self, msg: Message):
        """Handle broadcast messages."""
        event = msg.payload.get("event")
        logger.debug(f"Persona {self.agent_id} received broadcast: {event}")

        # Track other agents coming online
        if event == "agent_online":
            agent_id = msg.payload.get("agent_id")
            if agent_id and agent_id != self.agent_id:
                self._collaborators.add(agent_id)

    async def _handle_multicast(self, msg: Message):
        """Handle multicast/topic messages."""
        logger.debug(f"Persona {self.agent_id} received multicast from {msg.sender_id}")
        self._interactions += 1

    async def _handle_response(self, msg: Message):
        """Handle response messages."""
        logger.debug(f"Persona {self.agent_id} received response from {msg.sender_id}")

    async def _process_query(self, query: str) -> dict:
        """Process a query using persona knowledge."""
        # Simulate response generation
        # In a real implementation, this would use an LLM with the persona prompt

        response_text = f"Based on my expertise as defined in my persona, regarding '{query}': "
        response_text += "This requires careful consideration of the relevant factors."

        # Add to conversation history
        self._conversation_history.append({
            "role": "user",
            "content": query,
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        return {
            "status": "ok",
            "response": response_text,
            "confidence": random.uniform(0.5, 0.95),
        }

    def _get_relevant_knowledge(self, topic: str) -> list[dict]:
        """Get knowledge relevant to a topic."""
        return [
            k for k in self._knowledge_base
            if topic.lower() in k.get("topic", "").lower()
        ]

    def _update_belief(self, topic: str, evidence: dict):
        """Update belief using Bayesian-style update."""
        prior = self._beliefs.get(topic, 0.5)
        strength = evidence.get("strength", 0.5)
        direction = evidence.get("direction", 1)  # 1 for supporting, -1 for contradicting

        # Simple belief update
        if direction > 0:
            posterior = prior + (1 - prior) * strength * 0.3
        else:
            posterior = prior - prior * strength * 0.3

        self._beliefs[topic] = max(0.01, min(0.99, posterior))

    async def _execute_task(self, task: dict) -> dict:
        """Execute a delegated task."""
        task_type = task.get("type", "unknown")

        if task_type == "research":
            # Simulate research
            await asyncio.sleep(random.uniform(0.1, 0.5))
            return {
                "status": "completed",
                "findings": f"Research on {task.get('topic', 'unknown')} completed.",
                "confidence": random.uniform(0.6, 0.9),
            }

        elif task_type == "analyze":
            await asyncio.sleep(random.uniform(0.1, 0.3))
            return {
                "status": "completed",
                "analysis": f"Analysis of {task.get('subject', 'data')} complete.",
                "insights": ["insight_1", "insight_2"],
            }

        return {"status": "error", "message": f"Unknown task type: {task_type}"}

    def _evaluate_proposal(self, proposal: dict) -> dict:
        """Evaluate a proposal based on persona beliefs."""
        # Simple evaluation based on alignment with beliefs
        topic = proposal.get("topic", "")
        belief = self._beliefs.get(topic, 0.5)

        # Random variation + belief influence
        score = random.gauss(0.5, 0.2) + (belief - 0.5) * 0.3
        score = max(0, min(1, score))

        decision = "approve" if score > 0.5 else "reject"
        confidence = abs(score - 0.5) * 2

        return {
            "decision": decision,
            "rationale": f"Based on my expertise and beliefs (confidence in topic: {belief:.2f})",
            "confidence": confidence,
        }

    # Public API
    def add_knowledge(self, topic: str, content: str, source: Optional[str] = None):
        """Add to the agent's knowledge base."""
        self._knowledge_base.append({
            "topic": topic,
            "content": content,
            "source": source,
        })

    def set_belief(self, topic: str, confidence: float):
        """Set initial belief about a topic."""
        self._beliefs[topic] = max(0.01, min(0.99, confidence))

    def get_beliefs(self) -> dict[str, float]:
        """Get current beliefs."""
        return self._beliefs.copy()
