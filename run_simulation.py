#!/usr/bin/env python3
"""
Multi-Agent System Simulation Runner

Demonstrates broadcast, multicast, and direct messaging patterns
with beautiful Rich terminal output.
"""
from __future__ import annotations

import asyncio
import logging
import random
import sys
from datetime import datetime, timezone
from typing import Optional

# Setup path for local imports
sys.path.insert(0, ".")

from src.messaging import MessageBroker, BrokerConfig, MessageType, Priority
from src.agents import DemiurgeAgent, PersonaAgent, WorkerAgent
from src.core import Supervisor, AgentSpec, RestartPolicy, SupervisorConfig
from src.ui import Console, Dashboard, Theme

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Quiet logging, we use Rich console instead
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# UI
console = Console()
dashboard: Optional[Dashboard] = None


class SimulationOrchestrator:
    """
    Orchestrates the multi-agent simulation with styled output.
    """

    def __init__(
        self,
        use_dashboard: bool = False,
        num_personas: int = 3,
        num_workers: int = 2,
    ):
        self.use_dashboard = use_dashboard
        self.num_personas = num_personas
        self.num_workers = num_workers

        # Components
        self.broker: Optional[MessageBroker] = None
        self.supervisor: Optional[Supervisor] = None

        # Stats tracking
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "broadcasts": 0,
            "multicasts": 0,
            "direct_messages": 0,
            "errors": 0,
        }

        # Agent tracking
        self.agents: dict[str, str] = {}  # agent_id -> type

    async def setup(self):
        """Initialize the simulation environment."""
        console.header(
            "LIDA Multi-Agent System",
            subtitle="Initializing simulation environment..."
        )

        # Create broker
        console.info("Connecting to Redis...", source="broker")
        broker_config = BrokerConfig(redis_url="redis://localhost:6379")
        self.broker = MessageBroker(broker_config)

        try:
            await self.broker.connect()
            await self.broker.start()
            console.success("Redis connected", source="broker")
        except Exception as e:
            console.error(f"Redis connection failed: {e}", source="broker")
            console.warning("Continuing with in-memory messaging only", source="broker")

        # Create supervisor
        console.info("Creating supervisor...", source="supervisor")
        supervisor_config = SupervisorConfig(
            max_restarts=3,
            restart_window_seconds=60.0,
            health_check_interval_seconds=5.0,
        )
        self.supervisor = Supervisor(self.broker, supervisor_config, console)

        # Register agent specifications
        self._register_agent_specs()

        console.success("Supervisor initialized", source="supervisor")

    def _register_agent_specs(self):
        """Register agent types with the supervisor."""
        # Demiurge - the orchestrator
        self.supervisor.register_spec("demiurge", AgentSpec(
            agent_type="demiurge",
            agent_class=DemiurgeAgent,
            config={"agent_type": "demiurge", "agent_id": "demiurge-prime"},
            restart_policy=RestartPolicy.ALWAYS,
        ))

        # Persona agents
        for i in range(self.num_personas):
            self.supervisor.register_spec(f"persona_{i}", AgentSpec(
                agent_type="persona",
                agent_class=PersonaAgent,
                config={
                    "agent_type": "persona",
                    "agent_id": f"persona-{i}",
                },
                restart_policy=RestartPolicy.ON_FAILURE,
                dependencies=["demiurge-prime"],
                start_delay=0.2 * i,  # Stagger startup
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
        console.section("Spawning Agents")

        # Start supervisor
        await self.supervisor.start()

        # Spawn demiurge first
        console.info("Spawning Demiurge...", source="spawner")
        await self.supervisor.spawn("demiurge")
        self.agents["demiurge-prime"] = "demiurge"

        # Spawn personas
        console.info(f"Spawning {self.num_personas} Persona agents...", source="spawner")
        for i in range(self.num_personas):
            agent_id = await self.supervisor.spawn(f"persona_{i}")
            self.agents[agent_id] = "persona"
            await asyncio.sleep(0.1)  # Small delay between spawns

        # Spawn workers
        console.info(f"Spawning {self.num_workers} Worker agents...", source="spawner")
        for i in range(self.num_workers):
            agent_id = await self.supervisor.spawn(f"worker_{i}")
            self.agents[agent_id] = "worker"
            await asyncio.sleep(0.1)

        console.success(f"Spawned {len(self.agents)} agents", source="spawner")

        # Display agent table
        console.agents_table(self.supervisor.list_agents())

    async def run_scenarios(self, duration: float = 30.0):
        """Run demonstration scenarios."""
        console.section("Running Demonstration Scenarios")

        start_time = datetime.now(timezone.utc)
        scenario_interval = 3.0  # Run a scenario every 3 seconds

        scenarios = [
            self._scenario_broadcast,
            self._scenario_multicast,
            self._scenario_direct_message,
            self._scenario_delegate_work,
            self._scenario_consensus,
        ]

        scenario_idx = 0
        while (datetime.now(timezone.utc) - start_time).total_seconds() < duration:
            # Run next scenario
            scenario = scenarios[scenario_idx % len(scenarios)]
            try:
                await scenario()
            except Exception as e:
                console.error(f"Scenario failed: {e}", source="scenario")
                self.stats["errors"] += 1

            scenario_idx += 1

            # Update dashboard if active
            if dashboard:
                self._update_dashboard()

            # Wait before next scenario
            await asyncio.sleep(scenario_interval)

        console.info(f"Completed {scenario_idx} scenarios", source="runner")

    async def _scenario_broadcast(self):
        """Demonstrate broadcast messaging."""
        console.info("Executing BROADCAST scenario", source="scenario")

        # Get the demiurge
        demiurge = self.supervisor._agents.get("demiurge-prime")
        if not demiurge:
            return

        # Broadcast an announcement
        agent = demiurge.agent
        console.message(
            "broadcast",
            agent.agent_id,
            "ALL",
            summary="System announcement: health check"
        )

        await agent.broadcast(
            MessageType.BROADCAST,
            {
                "event": "system_announcement",
                "message": "Health check in progress",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        self.stats["broadcasts"] += 1
        self.stats["messages_sent"] += 1

    async def _scenario_multicast(self):
        """Demonstrate multicast/topic messaging."""
        console.info("Executing MULTICAST scenario", source="scenario")

        # Get a random persona
        personas = [
            state.agent for aid, state in self.supervisor._agents.items()
            if state.agent_type == "persona"
        ]
        if not personas:
            return

        sender = random.choice(personas)
        topic = random.choice(["discussions:general", "research:ai", "work:compute"])

        console.message(
            "multicast",
            sender.agent_id,
            f"topic:{topic.split(':')[1]}",
            summary=f"Publishing to {topic}"
        )

        await sender.publish(
            topic,
            {
                "type": "discussion",
                "content": f"Message from {sender.agent_id}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

        self.stats["multicasts"] += 1
        self.stats["messages_sent"] += 1

    async def _scenario_direct_message(self):
        """Demonstrate direct point-to-point messaging."""
        console.info("Executing DIRECT messaging scenario", source="scenario")

        # Get two random agents
        agent_ids = list(self.supervisor._agents.keys())
        if len(agent_ids) < 2:
            return

        sender_id, recipient_id = random.sample(agent_ids, 2)
        sender_state = self.supervisor._agents[sender_id]
        sender = sender_state.agent

        console.message(
            "direct",
            sender_id,
            recipient_id,
            summary="Direct request"
        )

        await sender.send(
            recipient_id,
            MessageType.REQUEST,
            {
                "action": "get_status",
                "from": sender_id,
            },
        )

        self.stats["direct_messages"] += 1
        self.stats["messages_sent"] += 1

    async def _scenario_delegate_work(self):
        """Demonstrate work delegation to workers."""
        console.info("Executing DELEGATE scenario", source="scenario")

        # Get demiurge and a worker
        demiurge_state = self.supervisor._agents.get("demiurge-prime")
        workers = [
            state for aid, state in self.supervisor._agents.items()
            if state.agent_type == "worker"
        ]

        if not demiurge_state or not workers:
            return

        worker = random.choice(workers)
        demiurge = demiurge_state.agent

        task_type = random.choice(["compute", "analysis", "io"])

        console.message(
            "direct",
            demiurge.agent_id,
            worker.agent_id,
            summary=f"Delegate {task_type} task"
        )

        await demiurge.send(
            worker.agent_id,
            MessageType.DELEGATE,
            {
                "task": {
                    "id": f"task-{random.randint(1000, 9999)}",
                    "type": task_type,
                    "data": [random.randint(1, 100) for _ in range(10)],
                    "duration": random.uniform(0.1, 1.0),
                }
            },
        )

        self.stats["messages_sent"] += 1

    async def _scenario_consensus(self):
        """Demonstrate consensus/voting."""
        console.info("Executing CONSENSUS scenario", source="scenario")

        # Get personas for voting
        personas = [
            state.agent for aid, state in self.supervisor._agents.items()
            if state.agent_type == "persona"
        ]

        if len(personas) < 2:
            return

        proposer = personas[0]
        voters = personas[1:]

        # Propose something
        proposal_id = f"proposal-{random.randint(1000, 9999)}"

        console.info(f"Proposal {proposal_id} initiated", source=proposer.agent_id)

        for voter in voters:
            console.message_flow(
                proposer.agent_id,
                voter.agent_id,
                "propose",
                direction="out"
            )

            await proposer.send(
                voter.agent_id,
                MessageType.PROPOSE,
                {
                    "id": proposal_id,
                    "topic": "system_upgrade",
                    "description": "Proposed enhancement to messaging",
                },
            )
            self.stats["messages_sent"] += 1

    def _update_dashboard(self):
        """Update dashboard with current state."""
        if not dashboard:
            return

        # Update agent states
        for aid, state in self.supervisor._agents.items():
            dashboard.update_agent(aid, {
                "id": aid,
                "type": state.agent_type,
                "status": state.status.value,
                "inbox": state.agent.inbound.pending_count() if hasattr(state.agent, "inbound") else 0,
                "outbox": state.agent.outbound.pending_count() if hasattr(state.agent, "outbound") else 0,
                "processed": state.agent.inbound.stats.processed if hasattr(state.agent, "inbound") else 0,
            })

        # Update stats
        dashboard.update_stats(self.stats)

    async def show_final_stats(self):
        """Display final statistics."""
        console.section("Final Statistics")

        # Collect stats from all agents
        total_inbox = 0
        total_outbox = 0
        total_processed = 0

        for aid, state in self.supervisor._agents.items():
            if hasattr(state.agent, "inbound"):
                total_inbox += state.agent.inbound.pending_count()
                total_processed += state.agent.inbound.stats.processed
            if hasattr(state.agent, "outbox"):
                total_outbox += state.agent.outbound.pending_count()

        self.stats["messages_received"] = total_processed

        console.stats_table(self.stats)
        console.print()

        # Show agent table
        console.agents_table(self.supervisor.list_agents())

    async def shutdown(self):
        """Gracefully shutdown the simulation."""
        console.section("Shutting Down")

        console.info("Terminating agents...", source="shutdown")
        await self.supervisor.stop()

        console.info("Disconnecting from Redis...", source="shutdown")
        if self.broker:
            await self.broker.stop()
            await self.broker.disconnect()

        console.success("Simulation complete", source="shutdown")

    async def run(self, duration: float = 30.0):
        """Run the full simulation."""
        try:
            await self.setup()
            await self.spawn_agents()

            if self.use_dashboard:
                # Run with live dashboard
                global dashboard
                dashboard = Dashboard()

                async def update_loop(_):
                    if self._running:
                        self._update_dashboard()

                self._running = True
                dashboard_task = asyncio.create_task(dashboard.run(update_loop))
                scenario_task = asyncio.create_task(self.run_scenarios(duration))

                await scenario_task
                dashboard.stop()
                await dashboard_task
            else:
                # Run with console output
                await self.run_scenarios(duration)

            await self.show_final_stats()

        except KeyboardInterrupt:
            console.warning("Interrupted by user", source="runner")
        finally:
            await self.shutdown()


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="LIDA Multi-Agent Simulation")
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Use live dashboard instead of console output",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Simulation duration in seconds (default: 30)",
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

    orchestrator = SimulationOrchestrator(
        use_dashboard=args.dashboard,
        num_personas=args.personas,
        num_workers=args.workers,
    )

    await orchestrator.run(duration=args.duration)


if __name__ == "__main__":
    asyncio.run(main())
