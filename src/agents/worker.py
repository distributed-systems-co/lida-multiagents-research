"""Worker agent for task execution."""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..messaging import Agent, AgentConfig, MessageBroker, Message, MessageType, Priority

logger = logging.getLogger(__name__)


class WorkerAgent(Agent):
    """
    A worker agent that executes tasks delegated from other agents.
    Can be specialized for different work types.
    """

    def __init__(
        self,
        broker: MessageBroker,
        config: Optional[AgentConfig] = None,
        work_types: Optional[List[str]] = None,
        capacity: int = 5,
    ):
        config = config or AgentConfig(agent_type="worker")
        super().__init__(broker, config)

        self.work_types = work_types or ["general", "compute", "io"]
        self.capacity = capacity

        # Task tracking
        self._active_tasks: Dict[str, dict] = {}
        self._completed_tasks: List[dict] = []
        self._failed_tasks: List[dict] = []

        # Stats
        self._total_executed = 0
        self._total_time = 0.0

        # Register handlers
        self.inbound.register_handler(MessageType.DELEGATE, self._handle_delegate)
        self.inbound.register_handler(MessageType.REQUEST, self._handle_request)
        self.inbound.register_handler(MessageType.BROADCAST, self._handle_broadcast)
        self.inbound.register_handler(MessageType.MULTICAST, self._handle_multicast)
        self.inbound.register_handler(MessageType.RESPONSE, self._handle_response)

    async def on_start(self):
        """Initialize worker."""
        logger.info(f"Worker {self.agent_id} starting with capacity {self.capacity}...")

        # Subscribe to work topics
        for work_type in self.work_types:
            await self.subscribe(f"work:{work_type}")

        # Announce availability
        await self.broadcast(
            MessageType.BROADCAST,
            {
                "event": "worker_online",
                "agent_id": self.agent_id,
                "work_types": self.work_types,
                "capacity": self.capacity,
            },
        )

    async def on_stop(self):
        """Cleanup on shutdown."""
        logger.info(f"Worker {self.agent_id} shutting down with {len(self._active_tasks)} active tasks")

        # Complete or cancel active tasks
        for task_id, task in list(self._active_tasks.items()):
            await self._fail_task(task_id, "worker_shutdown")

        await self.broadcast(
            MessageType.BROADCAST,
            {"event": "worker_offline", "agent_id": self.agent_id},
        )

    async def on_message(self, msg: Message):
        """Handle generic messages."""
        logger.debug(f"Worker {self.agent_id} received {msg.msg_type} from {msg.sender_id}")

    async def _handle_broadcast(self, msg: Message):
        """Handle broadcast messages."""
        event = msg.payload.get("event")

        if event == "work_available":
            # Check if we can take the work
            if len(self._active_tasks) < self.capacity:
                work_type = msg.payload.get("work_type")
                if work_type in self.work_types:
                    # Claim the work
                    await self.send(
                        msg.sender_id,
                        MessageType.REQUEST,
                        {
                            "action": "claim_work",
                            "work_id": msg.payload.get("work_id"),
                            "worker_id": self.agent_id,
                        },
                    )

    async def _handle_multicast(self, msg: Message):
        """Handle multicast messages."""
        # Check if this is work for our types
        work_type = msg.payload.get("work_type")
        if work_type in self.work_types and len(self._active_tasks) < self.capacity:
            await self._handle_delegate(msg)

    async def _handle_response(self, msg: Message):
        """Handle response messages."""
        logger.debug(f"Worker {self.agent_id} received response from {msg.sender_id}")

    async def _handle_request(self, msg: Message):
        """Handle status/capability requests."""
        action = msg.payload.get("action")

        if action == "get_status":
            response = {
                "status": "ok",
                "active_tasks": len(self._active_tasks),
                "capacity": self.capacity,
                "available": self.capacity - len(self._active_tasks),
                "total_executed": self._total_executed,
                "avg_time": self._total_time / max(1, self._total_executed),
            }
        elif action == "get_capabilities":
            response = {
                "status": "ok",
                "work_types": self.work_types,
                "capacity": self.capacity,
            }
        else:
            response = {"status": "error", "message": f"Unknown action: {action}"}

        await self.send(
            msg.sender_id,
            MessageType.RESPONSE,
            response,
            correlation_id=msg.msg_id,
        )

    async def _handle_delegate(self, msg: Message):
        """Handle delegated work."""
        task = msg.payload.get("task", msg.payload)
        task_id = task.get("id", msg.msg_id)

        # Check capacity
        if len(self._active_tasks) >= self.capacity:
            await self.send(
                msg.sender_id,
                MessageType.NACK,
                {"task_id": task_id, "reason": "at_capacity"},
                correlation_id=msg.msg_id,
            )
            return

        # Check if we can handle this work type
        work_type = task.get("type", "general")
        if work_type not in self.work_types:
            await self.send(
                msg.sender_id,
                MessageType.NACK,
                {"task_id": task_id, "reason": "unsupported_type"},
                correlation_id=msg.msg_id,
            )
            return

        # Accept the task
        self._active_tasks[task_id] = {
            "task": task,
            "requester": msg.sender_id,
            "started_at": datetime.utcnow(),
        }

        # ACK receipt
        await self.send(
            msg.sender_id,
            MessageType.ACK,
            {"task_id": task_id, "status": "accepted"},
            correlation_id=msg.msg_id,
        )

        # Execute asynchronously
        asyncio.create_task(self._execute_task(task_id))

    async def _execute_task(self, task_id: str):
        """Execute a task."""
        if task_id not in self._active_tasks:
            return

        task_info = self._active_tasks[task_id]
        task = task_info["task"]
        requester = task_info["requester"]

        try:
            # Simulate work based on type
            work_type = task.get("type", "general")
            duration = task.get("duration", random.uniform(0.1, 2.0))

            result = await self._do_work(work_type, task, duration)

            # Update stats
            elapsed = (datetime.utcnow() - task_info["started_at"]).total_seconds()
            self._total_executed += 1
            self._total_time += elapsed

            # Report success
            await self.send(
                requester,
                MessageType.REPORT,
                {
                    "task_id": task_id,
                    "status": "completed",
                    "result": result,
                    "duration": elapsed,
                },
            )

            self._completed_tasks.append({
                "task_id": task_id,
                "task": task,
                "result": result,
                "duration": elapsed,
            })

        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self._fail_task(task_id, str(e))

        finally:
            self._active_tasks.pop(task_id, None)

    async def _do_work(self, work_type: str, task: dict, duration: float) -> dict:
        """Perform the actual work."""
        # Simulate different work types
        await asyncio.sleep(duration)

        if work_type == "compute":
            # Simulate computation
            data = task.get("data", [])
            result = {"sum": sum(data) if isinstance(data, list) else 0}

        elif work_type == "io":
            # Simulate I/O
            result = {"bytes_processed": random.randint(1000, 1000000)}

        elif work_type == "analysis":
            result = {
                "insights": [f"insight_{i}" for i in range(random.randint(1, 5))],
                "confidence": random.uniform(0.6, 0.95),
            }

        else:  # general
            result = {"output": f"Completed {work_type} task"}

        return result

    async def _fail_task(self, task_id: str, reason: str):
        """Handle task failure."""
        if task_id not in self._active_tasks:
            return

        task_info = self._active_tasks.pop(task_id)
        elapsed = (datetime.utcnow() - task_info["started_at"]).total_seconds()

        self._failed_tasks.append({
            "task_id": task_id,
            "task": task_info["task"],
            "reason": reason,
            "duration": elapsed,
        })

        await self.send(
            task_info["requester"],
            MessageType.REPORT,
            {
                "task_id": task_id,
                "status": "failed",
                "reason": reason,
                "duration": elapsed,
            },
        )

    # Public API
    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "failed_tasks": len(self._failed_tasks),
            "total_executed": self._total_executed,
            "average_time": self._total_time / max(1, self._total_executed),
            "capacity": self.capacity,
            "utilization": len(self._active_tasks) / self.capacity,
        }
