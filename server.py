#!/usr/bin/env python3
"""
LIDA Multi-Agent System Server

Runs the FastAPI server with real-time WebSocket dashboard
and the multi-agent simulation.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional

import uvicorn

# Setup path for local imports
sys.path.insert(0, ".")

from src.messaging import MessageBroker, BrokerConfig, MessageType, Priority
from src.agents import DemiurgeAgent, PersonaAgent, WorkerAgent
from src.core import Supervisor, AgentSpec, RestartPolicy, SupervisorConfig
from src.api.app import create_app
from src.prompts import PromptLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ServerOrchestrator:
    """
    Orchestrates the multi-agent system for the web server.
    """

    def __init__(
        self,
        num_personas: int = 3,
        num_workers: int = 2,
    ):
        self.num_personas = num_personas
        self.num_workers = num_workers

        # Components
        self.broker: Optional[MessageBroker] = None
        self.supervisor: Optional[Supervisor] = None

        # Prompt loader
        self.prompt_loader = PromptLoader()
        self.prompt_loader.load()
        logger.info(f"Loaded {self.prompt_loader.count()} prompts")

        # Assigned prompts for personas
        self.persona_prompts: dict[str, dict] = {}

        # Stats tracking
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts": 0,
            "multicasts": 0,
            "direct_messages": 0,
            "errors": 0,
        }

    async def setup(self):
        """Initialize the system."""
        logger.info("Initializing multi-agent system...")

        # Create broker
        broker_config = BrokerConfig(redis_url="redis://localhost:6379")
        self.broker = MessageBroker(broker_config)

        try:
            await self.broker.connect()
            await self.broker.start()
            logger.info("Redis connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Using in-memory messaging.")

        # Create supervisor
        supervisor_config = SupervisorConfig(
            max_restarts=3,
            restart_window_seconds=60.0,
            health_check_interval_seconds=5.0,
        )
        self.supervisor = Supervisor(self.broker, supervisor_config)

        # Register agent specifications
        self._register_agent_specs()

        logger.info("Supervisor initialized")

    def _register_agent_specs(self):
        """Register agent types with the supervisor."""
        # Demiurge - the orchestrator with real baseline prompt
        self.supervisor.register_spec("demiurge", AgentSpec(
            agent_type="demiurge",
            agent_class=DemiurgeAgent,
            config={"agent_type": "demiurge", "agent_id": "demiurge-prime"},
            restart_policy=RestartPolicy.ALWAYS,
        ))

        # Persona agents - each gets a unique prompt from the library
        # Spread prompts across different categories for diversity
        from src.prompts import PromptCategory
        all_prompts = self.prompt_loader.all_prompts()
        active_categories = [c for c in PromptCategory if c != PromptCategory.DEMIURGE]
        prompts_by_cat = {cat: self.prompt_loader.get_by_category(cat) for cat in active_categories}
        # Filter to non-empty categories
        prompts_by_cat = {k: v for k, v in prompts_by_cat.items() if v}
        cat_list = list(prompts_by_cat.keys())

        for i in range(self.num_personas):
            # Round-robin across categories for diverse personas
            cat = cat_list[i % len(cat_list)]
            cat_prompts = prompts_by_cat[cat]
            prompt_idx = (i // len(cat_list)) % len(cat_prompts)
            prompt = cat_prompts[prompt_idx]
            agent_id = f"persona-{i}"

            if prompt:
                self.persona_prompts[agent_id] = {
                    "id": prompt.id,
                    "text": prompt.text,
                    "category": prompt.category.value,
                    "subcategory": prompt.subcategory,
                    "tags": prompt.tags,
                }

            self.supervisor.register_spec(f"persona_{i}", AgentSpec(
                agent_type="persona",
                agent_class=PersonaAgent,
                config={
                    "agent_type": "persona",
                    "agent_id": agent_id,
                },
                kwargs={
                    "persona_prompt": prompt.text if prompt else None,
                    "prompt_id": prompt.id if prompt else None,
                    "prompt_category": prompt.category.value if prompt else None,
                    "prompt_subcategory": prompt.subcategory if prompt else None,
                },
                restart_policy=RestartPolicy.ON_FAILURE,
                dependencies=["demiurge-prime"],
                start_delay=0.2 * i,
            ))

        # Worker agents
        for i in range(self.num_workers):
            self.supervisor.register_spec(f"worker_{i}", AgentSpec(
                agent_type="worker",
                agent_class=WorkerAgent,
                config={
                    "agent_type": "worker",
                    "agent_id": f"worker-{i}",
                },
                restart_policy=RestartPolicy.ON_FAILURE,
                start_delay=0.1 * i,
            ))

    async def spawn_agents(self):
        """Spawn all agents."""
        await self.supervisor.start()

        # Spawn demiurge first
        logger.info("Spawning Demiurge...")
        await self.supervisor.spawn("demiurge")

        # Spawn personas
        logger.info(f"Spawning {self.num_personas} Persona agents...")
        for i in range(self.num_personas):
            await self.supervisor.spawn(f"persona_{i}")
            await asyncio.sleep(0.1)

        # Spawn workers
        logger.info(f"Spawning {self.num_workers} Worker agents...")
        for i in range(self.num_workers):
            await self.supervisor.spawn(f"worker_{i}")
            await asyncio.sleep(0.1)

        logger.info(f"Spawned {len(self.supervisor._agents)} agents")

    async def shutdown(self):
        """Gracefully shutdown."""
        logger.info("Shutting down...")

        if self.supervisor:
            await self.supervisor.stop()

        if self.broker:
            await self.broker.stop()
            await self.broker.disconnect()

        logger.info("Shutdown complete")


async def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    num_personas: int = 3,
    num_workers: int = 2,
):
    """Run the server with agents."""
    # Create orchestrator
    orchestrator = ServerOrchestrator(
        num_personas=num_personas,
        num_workers=num_workers,
    )

    # Setup and spawn agents
    await orchestrator.setup()
    await orchestrator.spawn_agents()

    # Create FastAPI app
    app = create_app(orchestrator=orchestrator)

    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )

    server = uvicorn.Server(config)

    try:
        logger.info(f"Starting server at http://{host}:{port}")
        logger.info(f"Dashboard available at http://localhost:{port}")
        await server.serve()
    finally:
        await orchestrator.shutdown()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LIDA Multi-Agent Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--personas",
        type=int,
        default=3,
        help="Number of persona agents (default: 3)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker agents (default: 2)",
    )

    args = parser.parse_args()

    asyncio.run(run_server(
        host=args.host,
        port=args.port,
        num_personas=args.personas,
        num_workers=args.workers,
    ))


if __name__ == "__main__":
    main()
