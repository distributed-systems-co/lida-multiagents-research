#!/usr/bin/env python3
"""
Worker pool runner for LIDA Multi-Agent System.
Spawns multiple WorkerAgent instances that process tasks via Redis.
"""
import asyncio
import argparse
import logging
import os
import signal
from typing import List, Optional

from src.messaging import MessageBroker, BrokerConfig
from src.agents.worker import WorkerAgent
from src.messaging.agent import AgentConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class WorkerPool:
    """Manages a pool of worker agents."""

    def __init__(
        self,
        num_workers: int = 4,
        redis_url: str = "redis://localhost:6379",
        work_types: Optional[List[str]] = None,
        capacity_per_worker: int = 5,
    ):
        self.num_workers = num_workers
        self.redis_url = redis_url
        self.work_types = work_types or ["general", "compute", "io", "analysis", "llm"]
        self.capacity_per_worker = capacity_per_worker

        self.broker: Optional[MessageBroker] = None
        self.workers: List[WorkerAgent] = []
        self._running = False

    async def start(self):
        """Start the worker pool."""
        logger.info(f"Starting worker pool with {self.num_workers} workers...")

        # Connect broker
        self.broker = MessageBroker(BrokerConfig(redis_url=self.redis_url))
        await self.broker.connect()
        await self.broker.start()

        # Spawn workers
        for i in range(self.num_workers):
            config = AgentConfig(
                agent_type="worker",
                agent_id=f"worker-{i:03d}",
            )
            worker = WorkerAgent(
                broker=self.broker,
                config=config,
                work_types=self.work_types,
                capacity=self.capacity_per_worker,
            )
            await worker.start()
            self.workers.append(worker)
            logger.info(f"Started {worker.agent_id} with capacity {self.capacity_per_worker}")

        self._running = True
        logger.info(f"Worker pool ready: {self.num_workers} workers, total capacity {self.num_workers * self.capacity_per_worker}")

    async def stop(self):
        """Stop all workers."""
        logger.info("Stopping worker pool...")
        self._running = False

        for worker in self.workers:
            await worker.stop()
            logger.info(f"Stopped {worker.agent_id}")

        await self.broker.stop()
        await self.broker.disconnect()
        logger.info("Worker pool stopped")

    async def run_forever(self):
        """Run until interrupted."""
        await self.start()

        # Print stats periodically
        while self._running:
            await asyncio.sleep(30)
            if self._running:
                self._print_stats()

    def _print_stats(self):
        """Print worker statistics."""
        total_active = 0
        total_completed = 0
        total_failed = 0

        for worker in self.workers:
            stats = worker.get_stats()
            total_active += stats["active_tasks"]
            total_completed += stats["completed_tasks"]
            total_failed += stats["failed_tasks"]

        logger.info(
            f"Pool stats: active={total_active}, completed={total_completed}, failed={total_failed}"
        )


async def main():
    parser = argparse.ArgumentParser(description="LIDA Worker Pool")
    parser.add_argument(
        "-n", "--num-workers",
        type=int,
        default=int(os.getenv("NUM_WORKERS", "4")),
        help="Number of workers to spawn (default: 4)",
    )
    parser.add_argument(
        "--redis-url",
        default=os.getenv("REDIS_URL", "redis://localhost:6379"),
        help="Redis connection URL",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=int(os.getenv("WORKER_CAPACITY", "5")),
        help="Task capacity per worker (default: 5)",
    )
    parser.add_argument(
        "--work-types",
        default=os.getenv("WORK_TYPES", "general,compute,io,analysis,llm"),
        help="Comma-separated work types to handle",
    )

    args = parser.parse_args()

    work_types = [t.strip() for t in args.work_types.split(",")]

    pool = WorkerPool(
        num_workers=args.num_workers,
        redis_url=args.redis_url,
        work_types=work_types,
        capacity_per_worker=args.capacity,
    )

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(pool.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await pool.run_forever()
    except asyncio.CancelledError:
        pass
    finally:
        if pool._running:
            await pool.stop()


if __name__ == "__main__":
    asyncio.run(main())
