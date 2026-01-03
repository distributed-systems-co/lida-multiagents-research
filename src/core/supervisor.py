"""Agent supervisor for lifecycle and fault management."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any, Type
from enum import Enum

from .types import AgentStatus
from .config import SupervisorConfig

logger = logging.getLogger(__name__)


class RestartPolicy(str, Enum):
    """Agent restart policies."""
    NEVER = "never"           # Don't restart
    ON_FAILURE = "on_failure" # Restart only on error
    ALWAYS = "always"         # Always restart
    TRANSIENT = "transient"   # Restart on unexpected termination


@dataclass
class AgentSpec:
    """Specification for spawning an agent."""
    agent_type: str
    agent_class: Type
    config: dict = field(default_factory=dict)
    kwargs: dict = field(default_factory=dict)  # Extra kwargs passed to agent __init__
    restart_policy: RestartPolicy = RestartPolicy.ON_FAILURE
    dependencies: List[str] = field(default_factory=list)
    start_delay: float = 0.0


@dataclass
class AgentState:
    """Runtime state of a supervised agent."""
    agent_id: str
    agent_type: str
    spec: AgentSpec
    agent: Any  # The actual agent instance
    status: AgentStatus = AgentStatus.INITIALIZING
    start_count: int = 0
    last_start: Optional[datetime] = None
    last_stop: Optional[datetime] = None
    last_error: Optional[str] = None
    restarts_in_window: int = 0
    restart_window_start: Optional[datetime] = None


class Supervisor:
    """
    Supervises agent lifecycle and handles failures.

    Features:
    - Automatic restart on failure
    - Restart rate limiting
    - Dependency ordering
    - Health monitoring
    - Graceful shutdown
    """

    def __init__(
        self,
        broker: Any,  # MessageBroker
        config: Optional[SupervisorConfig] = None,
        ui: Optional[Any] = None,  # Console for styled output
    ):
        self.broker = broker
        self.config = config or SupervisorConfig()
        self.ui = ui

        self._agents: Dict[str, AgentState] = {}
        self._specs: Dict[str, AgentSpec] = {}
        self._running = False
        self._health_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, List[Callable]] = {}

    def register_spec(self, name: str, spec: AgentSpec):
        """Register an agent specification."""
        self._specs[name] = spec

    def on(self, event: str, handler: Callable):
        """Register an event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _emit(self, event: str, **kwargs):
        """Emit an event to handlers."""
        for handler in self._event_handlers.get(event, []):
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(**kwargs)
                else:
                    handler(**kwargs)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

    async def spawn(self, spec_name: str, agent_id: Optional[str] = None) -> str:
        """Spawn an agent from a registered spec."""
        from src.messaging.agent import AgentConfig

        if spec_name not in self._specs:
            raise ValueError(f"Unknown spec: {spec_name}")

        spec = self._specs[spec_name]

        # Check dependencies
        for dep in spec.dependencies:
            if dep not in self._agents or self._agents[dep].status != AgentStatus.RUNNING:
                raise RuntimeError(f"Dependency {dep} not running")

        # Create agent config
        config_dict = spec.config.copy()
        if agent_id:
            config_dict["agent_id"] = agent_id

        # Convert dict to AgentConfig
        config = AgentConfig(**config_dict)

        # Pass any extra kwargs to agent constructor
        agent = spec.agent_class(self.broker, config, **spec.kwargs)
        aid = agent.agent_id

        state = AgentState(
            agent_id=aid,
            agent_type=spec.agent_type,
            spec=spec,
            agent=agent,
        )
        self._agents[aid] = state

        # Start with delay if specified
        if spec.start_delay > 0:
            await asyncio.sleep(spec.start_delay)

        await self._start_agent(state)
        return aid

    async def _start_agent(self, state: AgentState):
        """Start an agent and update state."""
        try:
            state.start_count += 1
            state.last_start = datetime.now(timezone.utc)
            state.status = AgentStatus.STARTING

            await state.agent.start()

            state.status = AgentStatus.RUNNING

            if self.ui:
                self.ui.agent_spawned(state.agent_id, state.agent_type)

            await self._emit("agent_started", agent_id=state.agent_id, agent_type=state.agent_type)

            logger.info(f"Started agent {state.agent_id}")

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.last_error = str(e)
            logger.error(f"Failed to start agent {state.agent_id}: {e}")
            await self._handle_failure(state)

    async def _stop_agent(self, state: AgentState, reason: str = "requested"):
        """Stop an agent and update state."""
        try:
            state.status = AgentStatus.STOPPING
            state.last_stop = datetime.now(timezone.utc)

            await state.agent.stop()

            state.status = AgentStatus.TERMINATED

            if self.ui:
                self.ui.agent_terminated(state.agent_id, state.agent_type)

            await self._emit("agent_stopped", agent_id=state.agent_id, reason=reason)

            logger.info(f"Stopped agent {state.agent_id} ({reason})")

        except Exception as e:
            state.status = AgentStatus.ERROR
            state.last_error = str(e)
            logger.error(f"Error stopping agent {state.agent_id}: {e}")

    async def _handle_failure(self, state: AgentState):
        """Handle agent failure according to restart policy."""
        spec = state.spec
        policy = spec.restart_policy

        if policy == RestartPolicy.NEVER:
            logger.info(f"Agent {state.agent_id} failed, not restarting (policy=never)")
            return

        # Check restart rate limiting
        now = datetime.now(timezone.utc)
        if state.restart_window_start:
            elapsed = (now - state.restart_window_start).total_seconds()
            if elapsed > self.config.restart_window_seconds:
                # Reset window
                state.restarts_in_window = 0
                state.restart_window_start = now
        else:
            state.restart_window_start = now

        if state.restarts_in_window >= self.config.max_restarts:
            logger.error(
                f"Agent {state.agent_id} exceeded max restarts "
                f"({self.config.max_restarts} in {self.config.restart_window_seconds}s)"
            )
            state.status = AgentStatus.DEAD
            await self._emit("agent_dead", agent_id=state.agent_id, reason="max_restarts")
            return

        # Schedule restart
        state.restarts_in_window += 1
        logger.info(
            f"Restarting agent {state.agent_id} in {self.config.restart_delay_seconds}s "
            f"(attempt {state.restarts_in_window}/{self.config.max_restarts})"
        )

        await asyncio.sleep(self.config.restart_delay_seconds)
        await self._start_agent(state)

    async def terminate(self, agent_id: str, reason: str = "requested"):
        """Terminate an agent."""
        if agent_id not in self._agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        state = self._agents[agent_id]
        await self._stop_agent(state, reason)

    async def terminate_all(self, reason: str = "shutdown"):
        """Terminate all agents in reverse dependency order."""
        # Sort by dependencies (reverse)
        ordered = self._dependency_order(reverse=True)

        for agent_id in ordered:
            if agent_id in self._agents:
                await self.terminate(agent_id, reason)

    def _dependency_order(self, reverse: bool = False) -> List[str]:
        """Get agents in dependency order."""
        result = []
        visited = set()

        def visit(aid: str):
            if aid in visited:
                return
            visited.add(aid)
            state = self._agents.get(aid)
            if state:
                for dep in state.spec.dependencies:
                    visit(dep)
                result.append(aid)

        for aid in self._agents:
            visit(aid)

        if reverse:
            result.reverse()
        return result

    async def start(self):
        """Start the supervisor."""
        self._running = True
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("Supervisor started")

    async def stop(self):
        """Stop the supervisor."""
        self._running = False
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        await self.terminate_all("supervisor_shutdown")
        logger.info("Supervisor stopped")

    async def _health_loop(self):
        """Periodic health checks."""
        while self._running:
            await asyncio.sleep(self.config.health_check_interval_seconds)

            for state in list(self._agents.values()):
                if state.status == AgentStatus.RUNNING:
                    # Check if agent is actually responsive
                    try:
                        if hasattr(state.agent, "inbound"):
                            # Check mailbox stats
                            pending = state.agent.inbound.pending_count()
                            if pending > state.agent.inbound.max_size * 0.9:
                                logger.warning(
                                    f"Agent {state.agent_id} inbox nearly full ({pending})"
                                )

                        if self.ui:
                            self.ui.agent_status(
                                state.agent_id,
                                state.status.value,
                                f"inbox={state.agent.inbound.pending_count()}"
                            )

                    except Exception as e:
                        logger.error(f"Health check failed for {state.agent_id}: {e}")
                        state.status = AgentStatus.ERROR
                        await self._handle_failure(state)

    def get_status(self) -> Dict[str, dict]:
        """Get status of all supervised agents."""
        return {
            aid: {
                "agent_id": state.agent_id,
                "agent_type": state.agent_type,
                "status": state.status.value,
                "start_count": state.start_count,
                "last_start": state.last_start.isoformat() if state.last_start else None,
                "last_error": state.last_error,
                "inbox": state.agent.inbound.pending_count() if hasattr(state.agent, "inbound") else 0,
                "outbox": state.agent.outbound.pending_count() if hasattr(state.agent, "outbound") else 0,
            }
            for aid, state in self._agents.items()
        }

    def list_agents(self) -> List[dict]:
        """List all agents with their stats."""
        return [
            {
                "id": state.agent_id,
                "type": state.agent_type,
                "status": state.status.value,
                "inbox": state.agent.inbound.pending_count() if hasattr(state.agent, "inbound") else 0,
                "outbox": state.agent.outbound.pending_count() if hasattr(state.agent, "outbound") else 0,
                "processed": state.agent.inbound.stats.processed if hasattr(state.agent, "inbound") else 0,
            }
            for state in self._agents.values()
        ]
