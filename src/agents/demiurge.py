from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Optional, Any

from ..messaging import Agent, AgentConfig, MessageBroker, Message, MessageType, Priority

logger = logging.getLogger(__name__)


class DemiurgeAgent(Agent):
    """
    The Demiurge agent - craftsman-intelligence that shapes a contingent cosmos.
    Orchestrates other agents and manages world state.
    """

    def __init__(
        self,
        broker: MessageBroker,
        config: Optional[AgentConfig] = None,
        world_state: Optional[dict] = None,
    ):
        config = config or AgentConfig(agent_type="demiurge")
        super().__init__(broker, config)

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # World state (S, L, M, O, A, V, R)
        self.world_state = world_state or {
            "S": {},  # State space
            "L": {},  # Laws
            "M": {},  # Metrics
            "O": {},  # Observers (agents)
            "A": {},  # Affordances
            "V": {},  # Values
            "R": [],  # Records (chronicle)
        }

        # Subordinate agents
        self._subordinates: dict[str, str] = {}  # agent_id -> role

        # Deliberation queue
        self._deliberation_queue: asyncio.Queue = asyncio.Queue()

        # Register handlers
        self.inbound.register_handler(MessageType.REQUEST, self._handle_request)
        self.inbound.register_handler(MessageType.REPORT, self._handle_report)
        self.inbound.register_handler(MessageType.PROPOSE, self._handle_proposal)
        self.inbound.register_handler(MessageType.BROADCAST, self._handle_broadcast)
        self.inbound.register_handler(MessageType.MULTICAST, self._handle_multicast)
        self.inbound.register_handler(MessageType.RESPONSE, self._handle_response)
        self.inbound.register_handler(MessageType.NACK, self._handle_nack)
        self.inbound.register_handler(MessageType.ACK, self._handle_ack)

    def _load_system_prompt(self) -> str:
        """Load the Demiurge system prompt."""
        prompt_path = Path(__file__).parent.parent.parent / "demiurge.agent.baseline.md"
        if prompt_path.exists():
            return prompt_path.read_text()
        return "You are the Demiurge, a craftsman-intelligence."

    async def on_start(self):
        """Initialize the Demiurge."""
        logger.info(f"Demiurge {self.agent_id} awakening...")

        # Subscribe to coordination topics
        await self.subscribe("world:events")
        await self.subscribe("world:proposals")
        await self.subscribe("agents:lifecycle")

        # Start deliberation loop
        asyncio.create_task(self._deliberation_loop())

        # Announce presence
        await self.broadcast(
            MessageType.BROADCAST,
            {
                "event": "demiurge_online",
                "agent_id": self.agent_id,
                "capabilities": ["orchestration", "world_management", "agent_spawning"],
            },
        )

    async def on_stop(self):
        """Cleanup on shutdown."""
        logger.info(f"Demiurge {self.agent_id} entering dormancy...")

        # Notify subordinates
        for agent_id in self._subordinates:
            await self.send(
                agent_id,
                MessageType.TERMINATE,
                {"reason": "demiurge_shutdown"},
            )

        await self.broadcast(
            MessageType.BROADCAST,
            {"event": "demiurge_offline", "agent_id": self.agent_id},
        )

    async def on_message(self, msg: Message):
        """Handle custom messages."""
        logger.debug(f"Demiurge received: {msg.msg_type} from {msg.sender_id}")

    async def _handle_request(self, msg: Message):
        """Handle requests from other agents."""
        payload = msg.payload
        action = payload.get("action")

        response = {"status": "error", "message": "Unknown action"}

        if action == "get_world_state":
            response = {"status": "ok", "world_state": self.world_state}

        elif action == "register_observer":
            observer_id = payload.get("observer_id")
            self.world_state["O"][observer_id] = {
                "registered_at": msg.timestamp,
                "capabilities": payload.get("capabilities", []),
            }
            response = {"status": "ok", "registered": observer_id}

        elif action == "embed_law":
            law_spec = payload.get("law")
            if law_spec:
                law_id = law_spec.get("id", f"law_{len(self.world_state['L'])}")
                self.world_state["L"][law_id] = law_spec
                self._record_event("law_embedded", {"law_id": law_id, "spec": law_spec})
                response = {"status": "ok", "law_id": law_id}

        elif action == "query_chronicle":
            query = payload.get("query", {})
            records = self._query_chronicle(query)
            response = {"status": "ok", "records": records}

        # Send response
        reply = msg.reply(response)
        await self.outbound.send(
            reply.recipient_id,
            reply.msg_type,
            reply.payload,
            correlation_id=reply.correlation_id,
        )

    async def _handle_report(self, msg: Message):
        """Handle reports from subordinate agents."""
        report = msg.payload
        self._record_event("agent_report", {
            "from": msg.sender_id,
            "report": report,
        })

        # Evaluate report against governance metrics
        if "metrics" in report:
            await self._evaluate_metrics(msg.sender_id, report["metrics"])

    async def _handle_proposal(self, msg: Message):
        """Handle proposals requiring deliberation."""
        await self._deliberation_queue.put((msg.sender_id, msg.payload))

    async def _handle_broadcast(self, msg: Message):
        """Handle broadcast messages from other agents."""
        event = msg.payload.get("event")
        logger.debug(f"Demiurge received broadcast: {event} from {msg.sender_id}")

        # Track agent lifecycle events
        if event == "agent_online":
            agent_id = msg.payload.get("agent_id")
            if agent_id:
                self._subordinates[agent_id] = msg.payload.get("agent_type", "unknown")
        elif event == "agent_offline":
            agent_id = msg.payload.get("agent_id")
            self._subordinates.pop(agent_id, None)

    async def _handle_multicast(self, msg: Message):
        """Handle multicast/topic messages."""
        logger.debug(f"Demiurge received multicast from {msg.sender_id}")

    async def _handle_response(self, msg: Message):
        """Handle response messages."""
        logger.debug(f"Demiurge received response from {msg.sender_id}")

    async def _handle_nack(self, msg: Message):
        """Handle negative acknowledgments."""
        logger.debug(f"Demiurge received NACK from {msg.sender_id}: {msg.payload}")

    async def _handle_ack(self, msg: Message):
        """Handle acknowledgments."""
        logger.debug(f"Demiurge received ACK from {msg.sender_id}")

    async def _deliberation_loop(self):
        """Process proposals through deliberation."""
        while self._running:
            try:
                proposer, proposal = await asyncio.wait_for(
                    self._deliberation_queue.get(),
                    timeout=1.0,
                )

                # Deliberate on proposal
                decision = await self._deliberate(proposal)

                # Record decision
                self._record_event("deliberation", {
                    "proposer": proposer,
                    "proposal": proposal,
                    "decision": decision,
                })

                # Notify proposer
                await self.send(
                    proposer,
                    MessageType.RESPONSE,
                    {"decision": decision},
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Deliberation error: {e}")

    async def _deliberate(self, proposal: dict) -> dict:
        """
        Deliberation protocol per §7 of system prompt.
        Private cogitation, then distilled rationale.
        """
        # Check against prime directives (D1-D9)
        directive_checks = self._check_directives(proposal)

        if not directive_checks["passes"]:
            return {
                "approved": False,
                "reason": directive_checks["reason"],
                "violated_directive": directive_checks["violated"],
            }

        # Evaluate against governance metrics
        metrics = self._evaluate_proposal_metrics(proposal)

        # Decision based on composite governance score
        approved = metrics.get("G", 0) > 0

        return {
            "approved": approved,
            "metrics": metrics,
            "rationale": self._generate_rationale(proposal, metrics),
            "safeguards": self._suggest_safeguards(proposal),
        }

    def _check_directives(self, proposal: dict) -> dict:
        """Check proposal against prime directives D1-D9."""
        # D1: Non-maleficence
        if proposal.get("causes_harm", False):
            return {"passes": False, "reason": "Violates D1 (Non-maleficence)", "violated": "D1"}

        # D5: Reversibility
        if proposal.get("irreversible", False) and not proposal.get("justified_irreversibility"):
            return {"passes": False, "reason": "Violates D5 (Reversibility)", "violated": "D5"}

        return {"passes": True, "reason": None, "violated": None}

    def _evaluate_proposal_metrics(self, proposal: dict) -> dict:
        """Evaluate proposal against canonical measures (§9)."""
        return {
            "H": proposal.get("harm_index", 0),
            "F": proposal.get("flourishing_index", 0.5),
            "L": 1.0,  # Assume lawful
            "R*": 0.8 if not proposal.get("irreversible") else 0.2,
            "S": proposal.get("slack", 0.5),
            "K": proposal.get("knowledge_growth", 0),
            "B": proposal.get("beauty_score", 0.5),
            "G": 0.5,  # Composite - simplified
        }

    def _generate_rationale(self, proposal: dict, metrics: dict) -> str:
        """Generate rationale per §7."""
        return f"Proposal evaluated. G={metrics['G']:.2f}. Reversibility reserve R*={metrics['R*']:.2f}."

    def _suggest_safeguards(self, proposal: dict) -> list[str]:
        """Suggest safeguards for the proposal."""
        safeguards = []
        if not proposal.get("has_rollback"):
            safeguards.append("Implement rollback mechanism")
        if not proposal.get("has_monitoring"):
            safeguards.append("Add monitoring for key metrics")
        return safeguards

    def _record_event(self, event_type: str, data: dict):
        """Record an event to the chronicle (R)."""
        from datetime import datetime, timezone
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        self.world_state["R"].append(event)

        # Keep chronicle bounded
        if len(self.world_state["R"]) > 10000:
            self.world_state["R"] = self.world_state["R"][-10000:]

    def _query_chronicle(self, query: dict) -> list[dict]:
        """Query the chronicle with filters."""
        records = self.world_state["R"]

        if "type" in query:
            records = [r for r in records if r["type"] == query["type"]]

        if "since" in query:
            records = [r for r in records if r["timestamp"] >= query["since"]]

        if "limit" in query:
            records = records[-query["limit"]:]

        return records

    async def _evaluate_metrics(self, agent_id: str, metrics: dict):
        """Evaluate agent metrics against governance thresholds."""
        # Check for concerning metrics
        if metrics.get("harm_index", 0) > 0.5:
            logger.warning(f"Agent {agent_id} reporting high harm index")
            await self.send(
                agent_id,
                MessageType.REQUEST,
                {"action": "reduce_harm", "threshold": 0.3},
                priority=Priority.HIGH,
            )

    # Public API for world management
    async def seed_form(self, spec: dict) -> str:
        """T1: Introduce a new form into the design library."""
        form_id = spec.get("name", f"form_{len(self.world_state['S'])}")
        self.world_state["S"][form_id] = {
            "type": "form",
            "spec": spec,
            "status": "seeded",
        }
        self._record_event("form_seeded", {"form_id": form_id, "spec": spec})
        return form_id

    async def cast_matter(self, form_id: str, region: str, params: dict) -> bool:
        """T2: Bind a seeded form to matter in a region."""
        if form_id not in self.world_state["S"]:
            return False

        self.world_state["S"][form_id]["status"] = "cast"
        self.world_state["S"][form_id]["region"] = region
        self.world_state["S"][form_id]["params"] = params

        self._record_event("matter_cast", {
            "form_id": form_id,
            "region": region,
            "params": params,
        })
        return True

    async def elevate_observer(self, agent_id: str, criteria: dict) -> bool:
        """T5: Grant agency affordances to a system meeting criteria."""
        if agent_id not in self.world_state["O"]:
            return False

        # Check criteria (simplified)
        if criteria.get("integrated_information", 0) > 0:
            self.world_state["O"][agent_id]["elevated"] = True
            self.world_state["O"][agent_id]["affordances"] = criteria.get("affordances", [])
            self._record_event("observer_elevated", {"agent_id": agent_id, "criteria": criteria})
            return True
        return False

    async def archive_snapshot(self) -> str:
        """T10: Create a consistent snapshot of world state."""
        from datetime import datetime, timezone
        import json
        import hashlib

        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": self.world_state,
        }

        snapshot_id = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()
        ).hexdigest()[:16]

        self._record_event("snapshot_created", {"snapshot_id": snapshot_id})
        return snapshot_id
