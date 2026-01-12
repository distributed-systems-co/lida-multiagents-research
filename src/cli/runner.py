#!/usr/bin/env python3
"""
CLI Daemon Runner for LIDA Multi-Agent Research Platform

Supports headless experiment execution with:
- YAML-driven configuration
- Progress monitoring via file/socket
- Checkpoint and resume
- Batch experiment queuing
- Real-time metrics export
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("lida.runner")


class ExperimentStatus(Enum):
    """Experiment lifecycle states."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    # Identity
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "unnamed_experiment"
    description: str = ""

    # Scenario
    scenario_file: str = ""
    topic: str = ""

    # Execution
    max_rounds: int = 10
    timeout_seconds: int = 3600
    checkpoint_interval: int = 5  # Rounds between checkpoints

    # Agents
    agent_count: int = 6
    live_mode: bool = True  # Use real LLM calls
    model_override: Optional[str] = None

    # Interventions
    interventions: List[Dict[str, Any]] = field(default_factory=list)
    pre_round_hooks: List[str] = field(default_factory=list)

    # Output
    output_dir: str = "experiment_results"
    save_transcripts: bool = True
    save_metrics: bool = True
    export_latex: bool = False

    # Replication
    random_seed: Optional[int] = None
    replications: int = 1

    @classmethod
    def from_yaml(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create from parsed YAML config."""
        # Extract relevant fields
        exp_config = config_dict.get("experiment", {})
        sim_config = config_dict.get("simulation", {})
        agent_config = config_dict.get("agents", {})

        return cls(
            experiment_id=exp_config.get("id", str(uuid.uuid4())[:8]),
            name=exp_config.get("name", "unnamed"),
            description=exp_config.get("description", ""),
            scenario_file=config_dict.get("_source_file", ""),
            topic=sim_config.get("topic", sim_config.get("auto_start_topic", "")),
            max_rounds=sim_config.get("max_rounds", 10),
            timeout_seconds=exp_config.get("timeout", 3600),
            checkpoint_interval=exp_config.get("checkpoint_interval", 5),
            agent_count=agent_config.get("count", 6),
            live_mode=sim_config.get("live_mode", True),
            model_override=agent_config.get("model_override"),
            interventions=exp_config.get("interventions", []),
            pre_round_hooks=exp_config.get("pre_round_hooks", []),
            output_dir=exp_config.get("output_dir", "experiment_results"),
            save_transcripts=exp_config.get("save_transcripts", True),
            save_metrics=exp_config.get("save_metrics", True),
            export_latex=exp_config.get("export_latex", False),
            random_seed=exp_config.get("random_seed"),
            replications=exp_config.get("replications", 1),
        )


@dataclass
class ExperimentMetrics:
    """Real-time metrics from experiment execution."""
    experiment_id: str
    current_round: int = 0
    total_rounds: int = 0
    elapsed_seconds: float = 0.0

    # Agent metrics
    agents_active: int = 0
    total_messages: int = 0

    # Position tracking
    positions: Dict[str, float] = field(default_factory=dict)  # agent_id -> position
    position_changes: int = 0

    # Consensus
    consensus_score: float = 0.0
    convergence: float = 0.0
    polarization: float = 0.0

    # LLM metrics
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_cost_usd: float = 0.0

    # Timestamps
    started_at: Optional[str] = None
    last_update: Optional[str] = None


@dataclass
class Checkpoint:
    """Serializable experiment checkpoint for resume."""
    experiment_id: str
    config: Dict[str, Any]
    round_number: int
    debate_state: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]
    metrics: Dict[str, Any]
    random_state: Any
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class ExperimentRunner:
    """
    Daemon-capable experiment runner.

    Supports:
    - Single experiment execution
    - Batch queue processing
    - Checkpoint/resume
    - Real-time progress reporting
    """

    def __init__(
        self,
        config: ExperimentConfig,
        progress_callback: Optional[Callable[[ExperimentMetrics], None]] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.checkpoint_dir = Path(checkpoint_dir or "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.status = ExperimentStatus.PENDING
        self.metrics = ExperimentMetrics(experiment_id=config.experiment_id)
        self.debate_engine = None
        self._stop_requested = False
        self._pause_requested = False

        # Import debate engine lazily
        self._engine_module = None

    def _get_engine(self):
        """Lazy import of debate engine."""
        if self._engine_module is None:
            try:
                from src.simulation.advanced_debate_engine import AdvancedDebateEngine
                self._engine_module = AdvancedDebateEngine
            except ImportError:
                logger.warning("Could not import AdvancedDebateEngine, using mock")
                self._engine_module = MockDebateEngine
        return self._engine_module

    async def run(self) -> Dict[str, Any]:
        """Execute the experiment."""
        self.status = ExperimentStatus.INITIALIZING
        self.metrics.started_at = datetime.now().isoformat()

        logger.info(f"Starting experiment: {self.config.name} ({self.config.experiment_id})")

        try:
            # Initialize
            await self._initialize()

            # Check for existing checkpoint
            checkpoint = self._load_checkpoint()
            if checkpoint:
                logger.info(f"Resuming from checkpoint at round {checkpoint.round_number}")
                await self._restore_checkpoint(checkpoint)

            self.status = ExperimentStatus.RUNNING

            # Main execution loop
            results = await self._run_debate_loop()

            self.status = ExperimentStatus.COMPLETED
            return results

        except asyncio.CancelledError:
            self.status = ExperimentStatus.CANCELLED
            logger.info("Experiment cancelled")
            raise
        except Exception as e:
            self.status = ExperimentStatus.FAILED
            logger.error(f"Experiment failed: {e}")
            raise
        finally:
            await self._cleanup()

    async def _initialize(self):
        """Initialize experiment components."""
        # Set random seed if specified
        if self.config.random_seed is not None:
            import random
            import numpy as np
            random.seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        # Load scenario config
        from .config_loader import ConfigLoader
        loader = ConfigLoader()

        if self.config.scenario_file:
            scenario = loader.load(self.config.scenario_file)
        else:
            scenario = {}

        # Initialize debate engine
        EngineClass = self._get_engine()
        self.debate_engine = EngineClass(
            topic=self.config.topic,
            agent_count=self.config.agent_count,
            live_mode=self.config.live_mode,
            model_override=self.config.model_override,
            scenario_config=scenario,
        )

        self.metrics.total_rounds = self.config.max_rounds
        self.metrics.agents_active = self.config.agent_count

    async def _run_debate_loop(self) -> Dict[str, Any]:
        """Main debate execution loop."""
        start_time = time.time()

        for round_num in range(self.metrics.current_round, self.config.max_rounds):
            # Check for stop/pause
            if self._stop_requested:
                break

            while self._pause_requested:
                await asyncio.sleep(0.5)

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.config.timeout_seconds:
                logger.warning(f"Experiment timeout after {elapsed:.1f}s")
                break

            # Execute round
            self.metrics.current_round = round_num + 1
            self.metrics.elapsed_seconds = elapsed

            logger.info(f"Round {round_num + 1}/{self.config.max_rounds}")

            # Apply any interventions for this round
            await self._apply_interventions(round_num)

            # Run round
            if self.debate_engine:
                round_result = await self.debate_engine.run_round()
                self._update_metrics_from_round(round_result)

            # Progress callback
            if self.progress_callback:
                self.progress_callback(self.metrics)

            # Checkpoint if needed
            if (round_num + 1) % self.config.checkpoint_interval == 0:
                await self._save_checkpoint()

        # Final metrics
        self.metrics.last_update = datetime.now().isoformat()

        # Collect results
        results = await self._collect_results()

        # Export if configured
        if self.config.export_latex:
            await self._export_latex(results)

        return results

    async def _apply_interventions(self, round_num: int):
        """Apply configured interventions for this round."""
        for intervention in self.config.interventions:
            trigger_round = intervention.get("round", -1)
            if trigger_round == round_num:
                intervention_type = intervention.get("type")
                target = intervention.get("target")
                content = intervention.get("content")

                logger.info(f"Applying intervention: {intervention_type} on {target}")

                if self.debate_engine and hasattr(self.debate_engine, "inject_intervention"):
                    await self.debate_engine.inject_intervention(
                        intervention_type=intervention_type,
                        target=target,
                        content=content
                    )

    def _update_metrics_from_round(self, round_result: Dict[str, Any]):
        """Update metrics from round results."""
        if not round_result:
            return

        self.metrics.total_messages += round_result.get("messages", 0)

        # Update positions
        if "positions" in round_result:
            old_positions = self.metrics.positions.copy()
            self.metrics.positions = round_result["positions"]

            # Count position changes
            for agent_id, new_pos in self.metrics.positions.items():
                old_pos = old_positions.get(agent_id, 0.5)
                if abs(new_pos - old_pos) > 0.1:
                    self.metrics.position_changes += 1

        # Update consensus metrics
        if "consensus" in round_result:
            self.metrics.consensus_score = round_result["consensus"]
        if "convergence" in round_result:
            self.metrics.convergence = round_result["convergence"]
        if "polarization" in round_result:
            self.metrics.polarization = round_result["polarization"]

        # Update LLM metrics
        if "llm_stats" in round_result:
            stats = round_result["llm_stats"]
            self.metrics.llm_calls += stats.get("calls", 0)
            self.metrics.llm_tokens_in += stats.get("tokens_in", 0)
            self.metrics.llm_tokens_out += stats.get("tokens_out", 0)
            self.metrics.llm_cost_usd += stats.get("cost_usd", 0.0)

    async def _save_checkpoint(self):
        """Save checkpoint for resume capability."""
        self.status = ExperimentStatus.CHECKPOINTING

        checkpoint = Checkpoint(
            experiment_id=self.config.experiment_id,
            config=asdict(self.config),
            round_number=self.metrics.current_round,
            debate_state=self.debate_engine.get_state() if self.debate_engine else {},
            agent_states={},  # Populated by engine
            metrics=asdict(self.metrics),
            random_state=None,  # TODO: Capture random state
        )

        checkpoint_file = self.checkpoint_dir / f"{self.config.experiment_id}_checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {checkpoint_file}")
        self.status = ExperimentStatus.RUNNING

    def _load_checkpoint(self) -> Optional[Checkpoint]:
        """Load existing checkpoint if available."""
        checkpoint_file = self.checkpoint_dir / f"{self.config.experiment_id}_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                data = json.load(f)
            return Checkpoint(**data)
        return None

    async def _restore_checkpoint(self, checkpoint: Checkpoint):
        """Restore state from checkpoint."""
        self.metrics.current_round = checkpoint.round_number
        if self.debate_engine and hasattr(self.debate_engine, "restore_state"):
            await self.debate_engine.restore_state(checkpoint.debate_state)

    async def _collect_results(self) -> Dict[str, Any]:
        """Collect final experiment results."""
        results = {
            "experiment_id": self.config.experiment_id,
            "name": self.config.name,
            "topic": self.config.topic,
            "status": self.status.value,
            "metrics": asdict(self.metrics),
            "config": asdict(self.config),
            "completed_at": datetime.now().isoformat(),
        }

        if self.debate_engine:
            results["debate_summary"] = self.debate_engine.get_summary()
            results["final_positions"] = self.debate_engine.get_positions()
            results["transcript"] = self.debate_engine.get_transcript()

        # Save results
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"{self.config.experiment_id}_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Results saved: {results_file}")
        return results

    async def _export_latex(self, results: Dict[str, Any]):
        """Export results to LaTeX format."""
        try:
            from src.exporters.paper_export import PaperExporter
            exporter = PaperExporter(results)
            exporter.export_all(Path(self.config.output_dir) / "latex")
        except ImportError:
            logger.warning("PaperExporter not available, skipping LaTeX export")

    async def _cleanup(self):
        """Clean up resources."""
        if self.debate_engine and hasattr(self.debate_engine, "cleanup"):
            await self.debate_engine.cleanup()

    def pause(self):
        """Pause experiment execution."""
        self._pause_requested = True
        self.status = ExperimentStatus.PAUSED
        logger.info("Experiment paused")

    def resume(self):
        """Resume paused experiment."""
        self._pause_requested = False
        self.status = ExperimentStatus.RUNNING
        logger.info("Experiment resumed")

    def stop(self):
        """Request graceful stop."""
        self._stop_requested = True
        logger.info("Stop requested")


class MockDebateEngine:
    """Mock engine for testing without LLM calls."""

    def __init__(self, topic: str, agent_count: int, **kwargs):
        self.topic = topic
        self.agent_count = agent_count
        self.round = 0
        self.positions = {f"agent_{i}": 0.5 for i in range(agent_count)}
        self.transcript = []

    async def run_round(self) -> Dict[str, Any]:
        """Simulate a round."""
        self.round += 1

        # Random position drift
        import random
        for agent_id in self.positions:
            self.positions[agent_id] += random.uniform(-0.1, 0.1)
            self.positions[agent_id] = max(0, min(1, self.positions[agent_id]))

        # Simulate message
        self.transcript.append({
            "round": self.round,
            "speaker": f"agent_{random.randint(0, self.agent_count-1)}",
            "content": f"[Simulated message for round {self.round}]"
        })

        return {
            "messages": 1,
            "positions": self.positions.copy(),
            "consensus": 0.5,
            "convergence": abs(0.5 - sum(self.positions.values()) / len(self.positions)),
        }

    def get_state(self) -> Dict[str, Any]:
        return {"round": self.round, "positions": self.positions}

    async def restore_state(self, state: Dict[str, Any]):
        self.round = state.get("round", 0)
        self.positions = state.get("positions", self.positions)

    def get_summary(self) -> Dict[str, Any]:
        return {"total_rounds": self.round, "final_positions": self.positions}

    def get_positions(self) -> Dict[str, float]:
        return self.positions

    def get_transcript(self) -> List[Dict[str, Any]]:
        return self.transcript

    async def cleanup(self):
        pass


class BatchRunner:
    """Run multiple experiments in sequence or parallel."""

    def __init__(self, config_files: List[str], parallel: int = 1):
        self.config_files = config_files
        self.parallel = parallel
        self.results = []

    async def run_all(self) -> List[Dict[str, Any]]:
        """Execute all experiments."""
        from .config_loader import ConfigLoader
        loader = ConfigLoader()

        if self.parallel == 1:
            # Sequential execution
            for config_file in self.config_files:
                config_dict = loader.load(config_file)
                config = ExperimentConfig.from_yaml(config_dict)
                config.scenario_file = config_file

                runner = ExperimentRunner(config)
                result = await runner.run()
                self.results.append(result)
        else:
            # Parallel execution
            semaphore = asyncio.Semaphore(self.parallel)

            async def run_with_limit(config_file: str):
                async with semaphore:
                    config_dict = loader.load(config_file)
                    config = ExperimentConfig.from_yaml(config_dict)
                    config.scenario_file = config_file

                    runner = ExperimentRunner(config)
                    return await runner.run()

            tasks = [run_with_limit(cf) for cf in self.config_files]
            self.results = await asyncio.gather(*tasks, return_exceptions=True)

        return self.results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LIDA Experiment Runner - CLI daemon for multi-agent deliberation experiments"
    )

    parser.add_argument(
        "scenario",
        nargs="?",
        help="Scenario YAML file to run"
    )
    parser.add_argument(
        "--batch",
        nargs="+",
        help="Run multiple scenarios in batch"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel experiments (for batch mode)"
    )
    parser.add_argument(
        "--topic",
        help="Override debate topic"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override max rounds"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Experiment timeout in seconds"
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable LLM calls (simulation mode)"
    )
    parser.add_argument(
        "--output",
        default="experiment_results",
        help="Output directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--resume",
        help="Resume from experiment ID"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (detach from terminal)"
    )
    parser.add_argument(
        "--progress-file",
        help="Write progress to file (for daemon mode)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle daemon mode
    if args.daemon:
        # Fork and detach
        if os.fork() > 0:
            sys.exit(0)
        os.setsid()
        if os.fork() > 0:
            sys.exit(0)

        # Redirect stdout/stderr
        sys.stdout = open(args.progress_file or "/dev/null", "w")
        sys.stderr = sys.stdout

    # Setup progress callback
    progress_callback = None
    if args.progress_file:
        def write_progress(metrics: ExperimentMetrics):
            with open(args.progress_file, "w") as f:
                json.dump(asdict(metrics), f, indent=2)
        progress_callback = write_progress

    # Handle signals
    loop = asyncio.new_event_loop()
    runner = None

    def signal_handler(sig, frame):
        if runner:
            runner.stop()
        loop.stop()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if args.batch:
            # Batch mode
            batch = BatchRunner(args.batch, parallel=args.parallel)
            results = loop.run_until_complete(batch.run_all())
            print(f"Completed {len(results)} experiments")

        elif args.scenario:
            # Single experiment
            from .config_loader import ConfigLoader
            loader = ConfigLoader()
            config_dict = loader.load(args.scenario)
            config = ExperimentConfig.from_yaml(config_dict)
            config.scenario_file = args.scenario

            # Apply CLI overrides
            if args.topic:
                config.topic = args.topic
            if args.rounds:
                config.max_rounds = args.rounds
            if args.no_live:
                config.live_mode = False
            config.timeout_seconds = args.timeout
            config.output_dir = args.output

            # Resume if specified
            if args.resume:
                config.experiment_id = args.resume

            runner = ExperimentRunner(
                config,
                progress_callback=progress_callback,
                checkpoint_dir=args.checkpoint_dir
            )

            result = loop.run_until_complete(runner.run())
            print(f"Experiment completed: {result['status']}")

        else:
            parser.print_help()
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
