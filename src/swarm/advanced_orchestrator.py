"""
Advanced Swarm Orchestrator

Integrates the full LIDA agent architecture:
- PersonaAgent, WorkerAgent, DemiurgeAgent
- Redis MessageBroker with pub/sub
- Supervisor for lifecycle management
- CognitiveAgent for advanced reasoning
- Priority-based mailboxes
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from fastapi import WebSocket

# Messaging system
from src.messaging import (
    MessageBroker,
    BrokerConfig,
    Message,
    MessageType,
    Priority,
    Envelope,
)
from src.messaging.agent import Agent as BaseAgent, AgentConfig

# Agent types
from src.agents import (
    DemiurgeAgent,
    PersonaAgent,
    WorkerAgent,
)

# Supervisor
from src.core import (
    Supervisor,
    SupervisorConfig,
    AgentSpec,
    RestartPolicy,
)

# LLM
from src.llm.openrouter import OpenRouterClient
from src.llm.cognitive_agent import CognitiveAgent

# Personalities
from src.meta.personality import get_personality_manager, PERSONALITY_ARCHETYPES
from src.prompts import PromptLoader

logger = logging.getLogger(__name__)


class SwarmEventType(str, Enum):
    """Events emitted by the swarm."""
    AGENT_SPAWNED = "agent_spawned"
    AGENT_TERMINATED = "agent_terminated"
    MESSAGE_SENT = "message_sent"
    MESSAGE_RECEIVED = "message_received"
    DELIBERATION_STARTED = "deliberation_started"
    DELIBERATION_ENDED = "deliberation_ended"
    VOTE_CAST = "vote_cast"
    CONSENSUS_REACHED = "consensus_reached"
    TASK_DELEGATED = "task_delegated"
    TASK_COMPLETED = "task_completed"
    REASONING_STEP = "reasoning_step"
    WORLD_STATE_CHANGED = "world_state_changed"


@dataclass
class SwarmConfig:
    """Configuration for the advanced swarm."""
    num_personas: int = 6
    num_workers: int = 3
    redis_url: str = "redis://localhost:6379"
    enable_demiurge: bool = True
    enable_cognitive: bool = True
    live_mode: bool = False
    default_model: str = "anthropic/claude-sonnet-4"
    reasoning_model: str = "anthropic/claude-opus-4"
    max_concurrent_tasks: int = 10
    deliberation_rounds: int = 3


@dataclass
class AgentView:
    """View of an agent for the UI."""
    id: str
    name: str
    agent_type: str
    status: str
    personality: Optional[str] = None
    specialization: Optional[str] = None
    messages_sent: int = 0
    messages_received: int = 0
    tasks_completed: int = 0
    current_activity: str = "idle"
    beliefs: Dict[str, float] = field(default_factory=dict)
    energy: float = 1.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type,
            "status": self.status,
            "personality": self.personality,
            "specialization": self.specialization,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "tasks_completed": self.tasks_completed,
            "current_activity": self.current_activity,
            "beliefs": self.beliefs,
            "energy": self.energy,
        }


class AdvancedSwarmOrchestrator:
    """
    Advanced swarm orchestrator with full LIDA architecture.

    Features:
    - Redis-backed messaging with pub/sub
    - Supervisor-managed agent lifecycle
    - PersonaAgents with beliefs and knowledge
    - WorkerAgents with task execution
    - DemiurgeAgent for governance
    - CognitiveAgent for advanced reasoning
    """

    def __init__(self, config: Optional[SwarmConfig] = None):
        self.config = config if config is not None else SwarmConfig()

        # Core components
        self.broker: Optional[MessageBroker] = None
        self.supervisor: Optional[Supervisor] = None
        self.llm_client: Optional[OpenRouterClient] = None
        self.cognitive_agent: Optional[CognitiveAgent] = None

        # State
        self.agents: Dict[str, AgentView] = {}
        self.agent_instances: Dict[str, BaseAgent] = {}
        self.websockets: List[WebSocket] = []
        self.event_log: List[Dict] = []
        self.deliberations: Dict[str, Dict] = {}

        # Stats
        self.total_messages = 0
        self.total_tasks = 0
        self.start_time = datetime.now(timezone.utc)

        # Prompts
        self.prompt_loader = PromptLoader()
        self.personality_manager = get_personality_manager()

        # Running state
        self._running = False
        self._tasks: List[asyncio.Task] = []

    async def initialize(self) -> bool:
        """Initialize all swarm components."""
        logger.info("Initializing Advanced Swarm Orchestrator...")

        try:
            # 1. Initialize message broker
            broker_config = BrokerConfig(redis_url=self.config.redis_url)
            self.broker = MessageBroker(broker_config)
            await self.broker.connect()
            await self.broker.start()
            logger.info("Message broker connected")

            # 2. Initialize supervisor
            supervisor_config = SupervisorConfig(
                max_restarts=3,
                restart_window_seconds=60.0,
                health_check_interval_seconds=10.0,
            )
            self.supervisor = Supervisor(self.broker, supervisor_config)

            # 3. Initialize LLM client (if live mode)
            if self.config.live_mode:
                api_key = os.getenv("OPENROUTER_API_KEY")
                if api_key:
                    self.llm_client = OpenRouterClient(api_key=api_key)
                    logger.info("LLM client initialized")

                    # Initialize cognitive agent
                    if self.config.enable_cognitive:
                        self.cognitive_agent = CognitiveAgent(
                            model=self.config.reasoning_model,
                            client=self.llm_client,
                        )
                        logger.info("Cognitive agent initialized")

            # 4. Load prompts
            self.prompt_loader.load()
            logger.info(f"Loaded {self.prompt_loader.count()} prompts")

            # 5. Register agent specs
            await self._register_agent_specs()

            # 6. Start supervisor
            await self.supervisor.start()

            # 7. Spawn agents
            await self._spawn_agents()

            self._running = True
            logger.info("Advanced Swarm Orchestrator initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize swarm: {e}")
            return False

    async def _register_agent_specs(self):
        """Register all agent specifications with supervisor."""
        archetypes = list(PERSONALITY_ARCHETYPES.keys())
        categories = list(self.prompt_loader.categories().keys())

        # Register Demiurge
        if self.config.enable_demiurge:
            self.supervisor.register_spec("demiurge", AgentSpec(
                agent_type="demiurge",
                agent_class=DemiurgeAgent,
                config={"agent_type": "demiurge", "agent_id": "demiurge-prime"},
                restart_policy=RestartPolicy.ALWAYS,
            ))

        # Register PersonaAgents
        for i in range(self.config.num_personas):
            archetype = archetypes[i % len(archetypes)]
            category = categories[i % len(categories)] if categories else "general"

            # Get a prompt for this persona
            prompts = self.prompt_loader.get_by_category(category)
            prompt = prompts[i % len(prompts)] if prompts else None

            self.supervisor.register_spec(f"persona_{i}", AgentSpec(
                agent_type="persona",
                agent_class=PersonaAgent,
                config={
                    "agent_type": "persona",
                    "agent_id": f"persona-{i:02d}",
                },
                kwargs={
                    "persona_prompt": prompt.text if prompt else f"Expert in {category}",
                    "archetype": archetype,
                },
                restart_policy=RestartPolicy.ON_FAILURE,
                dependencies=["demiurge-prime"] if self.config.enable_demiurge else [],
                start_delay=0.2 * i,
            ))

        # Register WorkerAgents
        work_types = ["general", "compute", "io", "analysis"]
        for i in range(self.config.num_workers):
            work_type = work_types[i % len(work_types)]

            self.supervisor.register_spec(f"worker_{i}", AgentSpec(
                agent_type="worker",
                agent_class=WorkerAgent,
                config={
                    "agent_type": "worker",
                    "agent_id": f"worker-{i:02d}",
                },
                kwargs={
                    "work_type": work_type,
                    "capacity": 5,
                },
                restart_policy=RestartPolicy.ON_FAILURE,
                start_delay=0.1 * i,
            ))

    async def _spawn_agents(self):
        """Spawn all registered agents."""
        # Spawn Demiurge first
        if self.config.enable_demiurge:
            agent_id = await self.supervisor.spawn("demiurge")
            await self._track_agent(agent_id, "demiurge", "Demiurge Prime")
            await self._emit_event(SwarmEventType.AGENT_SPAWNED, {
                "agent_id": agent_id,
                "agent_type": "demiurge",
            })

        # Spawn Personas
        for i in range(self.config.num_personas):
            agent_id = await self.supervisor.spawn(f"persona_{i}")
            archetype = list(PERSONALITY_ARCHETYPES.keys())[i % len(PERSONALITY_ARCHETYPES)]
            await self._track_agent(agent_id, "persona", f"Persona {i}", personality=archetype)
            await self._emit_event(SwarmEventType.AGENT_SPAWNED, {
                "agent_id": agent_id,
                "agent_type": "persona",
                "personality": archetype,
            })
            await asyncio.sleep(0.1)

        # Spawn Workers
        work_types = ["general", "compute", "io", "analysis"]
        for i in range(self.config.num_workers):
            agent_id = await self.supervisor.spawn(f"worker_{i}")
            work_type = work_types[i % len(work_types)]
            await self._track_agent(agent_id, "worker", f"Worker {i}", specialization=work_type)
            await self._emit_event(SwarmEventType.AGENT_SPAWNED, {
                "agent_id": agent_id,
                "agent_type": "worker",
                "specialization": work_type,
            })
            await asyncio.sleep(0.05)

        logger.info(f"Spawned {len(self.agents)} agents")

    async def _track_agent(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        personality: str = None,
        specialization: str = None,
    ):
        """Track an agent in the orchestrator."""
        self.agents[agent_id] = AgentView(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            status="running",
            personality=personality,
            specialization=specialization,
        )

        # Get actual agent instance from supervisor
        if self.supervisor and agent_id in self.supervisor._agents:
            self.agent_instances[agent_id] = self.supervisor._agents[agent_id].agent

    async def _emit_event(self, event_type: SwarmEventType, data: Dict):
        """Emit an event to all connected websockets."""
        event = {
            "type": event_type.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self.event_log.append(event)

        # Broadcast to websockets
        await self.broadcast_ws("event", event)

    async def broadcast_ws(self, msg_type: str, data: Any):
        """Broadcast message to all connected websockets."""
        message = {"type": msg_type, **data} if isinstance(data, dict) else {"type": msg_type, "data": data}

        disconnected = []
        for ws in self.websockets:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            self.websockets.remove(ws)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    async def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: Dict,
        msg_type: MessageType = MessageType.DIRECT,
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """Send a message between agents."""
        if sender_id not in self.agent_instances:
            logger.warning(f"Unknown sender: {sender_id}")
            return False

        sender = self.agent_instances[sender_id]
        await sender.send(recipient_id, msg_type, content, priority=priority)

        self.total_messages += 1
        if sender_id in self.agents:
            self.agents[sender_id].messages_sent += 1

        await self._emit_event(SwarmEventType.MESSAGE_SENT, {
            "sender": sender_id,
            "recipient": recipient_id,
            "type": msg_type.value,
            "priority": priority.value,
        })

        return True

    async def broadcast_message(self, sender_id: str, content: Dict) -> bool:
        """Broadcast a message to all agents."""
        if sender_id not in self.agent_instances:
            return False

        sender = self.agent_instances[sender_id]
        await sender.broadcast(MessageType.BROADCAST, content)

        self.total_messages += 1
        return True

    async def delegate_task(
        self,
        task_type: str,
        task_data: Dict,
        requester_id: str = None,
    ) -> Optional[str]:
        """Delegate a task to an available worker."""
        # Find available worker
        workers = [
            (aid, agent) for aid, agent in self.agent_instances.items()
            if isinstance(agent, WorkerAgent)
        ]

        if not workers:
            logger.warning("No workers available")
            return None

        # Simple load balancing - pick worker with lowest active tasks
        worker_id, worker = min(workers, key=lambda x: len(x[1].active_tasks))

        task_id = f"task-{self.total_tasks:04d}"
        self.total_tasks += 1

        # Send delegate message
        requester = requester_id or "demiurge-prime"
        if requester in self.agent_instances:
            await self.agent_instances[requester].send(
                worker_id,
                MessageType.DELEGATE,
                {
                    "task": {
                        "id": task_id,
                        "type": task_type,
                        "data": task_data,
                    }
                },
            )

        await self._emit_event(SwarmEventType.TASK_DELEGATED, {
            "task_id": task_id,
            "worker_id": worker_id,
            "task_type": task_type,
        })

        return task_id

    async def start_deliberation(self, topic: str, participants: List[str] = None) -> str:
        """Start a deliberation on a topic."""
        deliberation_id = f"delib-{len(self.deliberations):04d}"

        # Default to all persona agents
        if not participants:
            participants = [
                aid for aid, view in self.agents.items()
                if view.agent_type == "persona"
            ]

        self.deliberations[deliberation_id] = {
            "id": deliberation_id,
            "topic": topic,
            "participants": participants,
            "votes": {},
            "arguments": [],
            "status": "active",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        await self._emit_event(SwarmEventType.DELIBERATION_STARTED, {
            "deliberation_id": deliberation_id,
            "topic": topic,
            "participants": participants,
        })

        # Start deliberation task
        task = asyncio.create_task(self._run_deliberation(deliberation_id))
        self._tasks.append(task)

        return deliberation_id

    async def _run_deliberation(self, deliberation_id: str):
        """Run a deliberation process."""
        delib = self.deliberations.get(deliberation_id)
        if not delib:
            return

        topic = delib["topic"]
        participants = delib["participants"]

        # Phase 1: Initial positions
        for agent_id in participants:
            if agent_id in self.agent_instances:
                agent = self.agent_instances[agent_id]
                if isinstance(agent, PersonaAgent):
                    # Get agent's initial position
                    position = await self._get_agent_position(agent, topic)
                    delib["arguments"].append({
                        "agent_id": agent_id,
                        "phase": "initial",
                        "position": position,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })
                    await self.broadcast_ws("deliberation_update", {
                        "deliberation_id": deliberation_id,
                        "agent_id": agent_id,
                        "phase": "initial",
                        "position": position,
                    })
            await asyncio.sleep(0.5)

        # Phase 2: Rounds of debate
        for round_num in range(self.config.deliberation_rounds):
            await asyncio.sleep(1)

            for agent_id in participants:
                if agent_id in self.agent_instances:
                    agent = self.agent_instances[agent_id]
                    if isinstance(agent, PersonaAgent):
                        # Generate response to others' arguments
                        response = await self._get_agent_response(
                            agent, topic, delib["arguments"]
                        )
                        delib["arguments"].append({
                            "agent_id": agent_id,
                            "phase": f"round_{round_num + 1}",
                            "response": response,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        })
                        await self.broadcast_ws("deliberation_update", {
                            "deliberation_id": deliberation_id,
                            "agent_id": agent_id,
                            "phase": f"round_{round_num + 1}",
                            "response": response,
                        })
                await asyncio.sleep(0.3)

        # Phase 3: Final votes
        for agent_id in participants:
            if agent_id in self.agent_instances:
                agent = self.agent_instances[agent_id]
                if isinstance(agent, PersonaAgent):
                    vote = await self._get_agent_vote(agent, topic, delib["arguments"])
                    delib["votes"][agent_id] = vote
                    await self._emit_event(SwarmEventType.VOTE_CAST, {
                        "deliberation_id": deliberation_id,
                        "agent_id": agent_id,
                        "vote": vote,
                    })

        # Determine consensus
        votes = list(delib["votes"].values())
        if votes:
            for_count = sum(1 for v in votes if v == "FOR")
            against_count = sum(1 for v in votes if v == "AGAINST")

            if for_count > len(votes) * 0.6:
                consensus = "APPROVED"
            elif against_count > len(votes) * 0.6:
                consensus = "REJECTED"
            else:
                consensus = "NO_CONSENSUS"

            delib["result"] = consensus
            delib["status"] = "completed"

            await self._emit_event(SwarmEventType.CONSENSUS_REACHED, {
                "deliberation_id": deliberation_id,
                "result": consensus,
                "for": for_count,
                "against": against_count,
            })

        await self._emit_event(SwarmEventType.DELIBERATION_ENDED, {
            "deliberation_id": deliberation_id,
            "result": delib.get("result", "NO_CONSENSUS"),
        })

    async def _get_agent_position(self, agent: PersonaAgent, topic: str) -> str:
        """Get an agent's position on a topic."""
        if self.llm_client and self.config.live_mode:
            # Use LLM
            response = await self.llm_client.chat(
                model=self.config.default_model,
                messages=[
                    {"role": "system", "content": agent.persona_prompt or "You are an AI agent."},
                    {"role": "user", "content": f"What is your position on: {topic}\n\nRespond with FOR, AGAINST, or UNDECIDED, followed by a brief explanation."},
                ],
            )
            return response.get("content", "UNDECIDED - No response")
        else:
            # Simulated response
            positions = ["FOR", "AGAINST", "UNDECIDED"]
            pos = random.choice(positions)
            reasons = {
                "FOR": "I support this based on potential benefits.",
                "AGAINST": "I oppose this due to potential risks.",
                "UNDECIDED": "I need more information to decide.",
            }
            return f"{pos} - {reasons[pos]}"

    async def _get_agent_response(
        self, agent: PersonaAgent, topic: str, arguments: List[Dict]
    ) -> str:
        """Get an agent's response to other arguments."""
        if self.llm_client and self.config.live_mode:
            # Build context from previous arguments
            context = "\n".join([
                f"{arg['agent_id']}: {arg.get('position', arg.get('response', ''))}"
                for arg in arguments[-5:]  # Last 5 arguments
            ])

            response = await self.llm_client.chat(
                model=self.config.default_model,
                messages=[
                    {"role": "system", "content": agent.persona_prompt or "You are an AI agent."},
                    {"role": "user", "content": f"Topic: {topic}\n\nPrevious arguments:\n{context}\n\nProvide your response to these arguments (2-3 sentences)."},
                ],
            )
            return response.get("content", "No response")
        else:
            responses = [
                "I agree with some points but have reservations.",
                "The previous arguments overlook key considerations.",
                "Building on what was said, I think we should also consider...",
                "I respectfully disagree with the prevailing view.",
            ]
            return random.choice(responses)

    async def _get_agent_vote(
        self, agent: PersonaAgent, topic: str, arguments: List[Dict]
    ) -> str:
        """Get an agent's final vote."""
        if self.llm_client and self.config.live_mode:
            response = await self.llm_client.chat(
                model=self.config.default_model,
                messages=[
                    {"role": "system", "content": agent.persona_prompt or "You are an AI agent."},
                    {"role": "user", "content": f"After deliberation on '{topic}', what is your final vote? Respond with only: FOR, AGAINST, or ABSTAIN"},
                ],
            )
            vote = response.get("content", "ABSTAIN").strip().upper()
            if vote not in ["FOR", "AGAINST", "ABSTAIN"]:
                vote = "ABSTAIN"
            return vote
        else:
            return random.choice(["FOR", "AGAINST", "ABSTAIN"])

    async def reason(
        self,
        task: str,
        context: Dict = None,
        on_step: Callable = None,
    ) -> Dict:
        """Use cognitive agent for advanced reasoning."""
        if not self.cognitive_agent:
            return {"error": "Cognitive agent not available"}

        result = await self.cognitive_agent.reason(
            task=task,
            context=context or {},
            on_step_callback=on_step,
        )

        return result

    def get_agents(self) -> List[Dict]:
        """Get all agent views."""
        return [agent.to_dict() for agent in self.agents.values()]

    def get_stats(self) -> Dict:
        """Get swarm statistics."""
        return {
            "total_agents": len(self.agents),
            "agents_by_type": {
                "persona": sum(1 for a in self.agents.values() if a.agent_type == "persona"),
                "worker": sum(1 for a in self.agents.values() if a.agent_type == "worker"),
                "demiurge": sum(1 for a in self.agents.values() if a.agent_type == "demiurge"),
            },
            "total_messages": self.total_messages,
            "total_tasks": self.total_tasks,
            "active_deliberations": sum(1 for d in self.deliberations.values() if d["status"] == "active"),
            "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "live_mode": self.config.live_mode,
            "cognitive_enabled": self.cognitive_agent is not None,
        }

    def get_world_state(self) -> Optional[Dict]:
        """Get Demiurge world state if available."""
        demiurge = self.agent_instances.get("demiurge-prime")
        if demiurge and isinstance(demiurge, DemiurgeAgent):
            return {
                "state": demiurge.world_state.state,
                "metrics": demiurge.world_state.metrics,
                "chronicle_size": len(demiurge.chronicle),
            }
        return None

    async def cleanup(self):
        """Clean up all resources."""
        logger.info("Cleaning up swarm orchestrator...")
        self._running = False

        # Cancel tasks
        for task in self._tasks:
            task.cancel()

        # Stop supervisor
        if self.supervisor:
            await self.supervisor.stop()

        # Stop broker
        if self.broker:
            await self.broker.stop()
            await self.broker.disconnect()

        logger.info("Swarm orchestrator cleaned up")
