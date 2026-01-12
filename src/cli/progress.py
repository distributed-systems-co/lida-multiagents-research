#!/usr/bin/env python3
"""
Progress Monitoring System for LIDA Experiments

Provides:
- Real-time experiment progress tracking
- Terminal UI with rich output
- File-based progress for daemon mode
- WebSocket/SSE broadcasting
- Metrics aggregation
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

logger = logging.getLogger("lida.progress")


@dataclass
class ExperimentProgress:
    """Current progress state of an experiment."""
    experiment_id: str
    name: str = ""
    status: str = "pending"

    # Progress
    current_round: int = 0
    total_rounds: int = 0
    percent_complete: float = 0.0

    # Timing
    started_at: Optional[str] = None
    elapsed_seconds: float = 0.0
    estimated_remaining: float = 0.0

    # Agents
    agents_total: int = 0
    agents_active: int = 0
    agents_speaking: List[str] = field(default_factory=list)

    # Messages
    total_messages: int = 0
    messages_this_round: int = 0

    # Positions
    positions: Dict[str, float] = field(default_factory=dict)
    position_changes: int = 0

    # Consensus
    consensus_score: float = 0.0
    majority_position: str = "undecided"

    # LLM Metrics
    llm_calls: int = 0
    llm_tokens: int = 0
    llm_cost_usd: float = 0.0

    # Current activity
    current_phase: str = ""
    current_speaker: str = ""
    last_message: str = ""

    # Errors
    errors: List[str] = field(default_factory=list)

    def update_percent(self):
        """Calculate percent complete."""
        if self.total_rounds > 0:
            self.percent_complete = (self.current_round / self.total_rounds) * 100

    def estimate_remaining(self):
        """Estimate remaining time based on current pace."""
        if self.current_round > 0 and self.elapsed_seconds > 0:
            avg_round_time = self.elapsed_seconds / self.current_round
            remaining_rounds = self.total_rounds - self.current_round
            self.estimated_remaining = avg_round_time * remaining_rounds


class ProgressReporter:
    """
    Reports experiment progress to various outputs.

    Supports:
    - File output (JSON) for daemon mode
    - WebSocket/SSE for real-time dashboards
    - Callbacks for programmatic access
    - Aggregated batch progress
    """

    def __init__(
        self,
        output_file: Optional[str] = None,
        websocket_url: Optional[str] = None,
        callback: Optional[Callable[[ExperimentProgress], None]] = None,
        update_interval: float = 1.0,
    ):
        self.output_file = Path(output_file) if output_file else None
        self.websocket_url = websocket_url
        self.callback = callback
        self.update_interval = update_interval

        self._experiments: Dict[str, ExperimentProgress] = {}
        self._websocket = None
        self._running = False
        self._update_task = None

    def register_experiment(self, progress: ExperimentProgress):
        """Register an experiment for tracking."""
        self._experiments[progress.experiment_id] = progress
        self._write_progress()

    def update(self, experiment_id: str, **kwargs):
        """Update experiment progress."""
        if experiment_id not in self._experiments:
            self._experiments[experiment_id] = ExperimentProgress(experiment_id=experiment_id)

        progress = self._experiments[experiment_id]

        for key, value in kwargs.items():
            if hasattr(progress, key):
                setattr(progress, key, value)

        progress.update_percent()
        progress.estimate_remaining()

        self._write_progress()

        if self.callback:
            self.callback(progress)

    def get_progress(self, experiment_id: str) -> Optional[ExperimentProgress]:
        """Get current progress for an experiment."""
        return self._experiments.get(experiment_id)

    def get_all_progress(self) -> Dict[str, ExperimentProgress]:
        """Get progress for all experiments."""
        return self._experiments.copy()

    def _write_progress(self):
        """Write progress to output file."""
        if not self.output_file:
            return

        try:
            data = {
                "updated_at": datetime.now().isoformat(),
                "experiments": {
                    exp_id: asdict(prog)
                    for exp_id, prog in self._experiments.items()
                }
            }

            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.warning(f"Failed to write progress: {e}")

    async def start_websocket_broadcast(self):
        """Start WebSocket broadcasting."""
        if not self.websocket_url:
            return

        try:
            import websockets
        except ImportError:
            logger.warning("websockets not installed, skipping WebSocket broadcast")
            return

        self._running = True

        async def broadcast_loop():
            while self._running:
                try:
                    async with websockets.connect(self.websocket_url) as ws:
                        self._websocket = ws
                        while self._running:
                            data = {
                                "type": "progress",
                                "timestamp": datetime.now().isoformat(),
                                "experiments": {
                                    exp_id: asdict(prog)
                                    for exp_id, prog in self._experiments.items()
                                }
                            }
                            await ws.send(json.dumps(data))
                            await asyncio.sleep(self.update_interval)

                except Exception as e:
                    logger.warning(f"WebSocket error: {e}")
                    await asyncio.sleep(5)  # Reconnect delay

        self._update_task = asyncio.create_task(broadcast_loop())

    async def stop(self):
        """Stop broadcasting."""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass


class TerminalUI:
    """
    Rich terminal UI for experiment progress.

    Features:
    - Live updating progress bars
    - Agent position visualization
    - Message stream
    - Metrics dashboard
    """

    def __init__(self, refresh_rate: float = 0.5):
        self.refresh_rate = refresh_rate
        self._running = False

        # Try to import rich for fancy output
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.table import Table
            from rich.panel import Panel
            from rich.progress import Progress, BarColumn, TextColumn
            from rich.layout import Layout
            self.rich_available = True
            self.console = Console()
        except ImportError:
            self.rich_available = False
            self.console = None

    def display_progress(self, progress: ExperimentProgress):
        """Display current progress in terminal."""
        if self.rich_available:
            self._display_rich(progress)
        else:
            self._display_simple(progress)

    def _display_simple(self, progress: ExperimentProgress):
        """Simple text-based progress display."""
        # Clear line and print status
        sys.stdout.write("\r" + " " * 80 + "\r")

        bar_width = 30
        filled = int(progress.percent_complete / 100 * bar_width)
        bar = "=" * filled + "-" * (bar_width - filled)

        status = (
            f"[{bar}] {progress.percent_complete:.1f}% | "
            f"Round {progress.current_round}/{progress.total_rounds} | "
            f"Messages: {progress.total_messages} | "
            f"Consensus: {progress.consensus_score:.2f}"
        )

        sys.stdout.write(status)
        sys.stdout.flush()

    def _display_rich(self, progress: ExperimentProgress):
        """Rich terminal display with panels and tables."""
        from rich.table import Table
        from rich.panel import Panel
        from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
        from rich import box

        # Clear screen
        self.console.clear()

        # Header
        self.console.print(
            Panel(
                f"[bold blue]{progress.name}[/] | "
                f"ID: {progress.experiment_id} | "
                f"Status: [green]{progress.status}[/]",
                title="LIDA Experiment Runner",
                border_style="blue"
            )
        )

        # Progress bar
        with Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            TextColumn("{task.fields[eta]}"),
            console=self.console,
            transient=True,
        ) as prog:
            task = prog.add_task(
                "Rounds",
                total=progress.total_rounds,
                completed=progress.current_round,
                eta=f"ETA: {progress.estimated_remaining:.0f}s"
            )

        # Metrics table
        metrics_table = Table(title="Metrics", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        metrics_table.add_row("Total Messages", str(progress.total_messages))
        metrics_table.add_row("Position Changes", str(progress.position_changes))
        metrics_table.add_row("Consensus Score", f"{progress.consensus_score:.3f}")
        metrics_table.add_row("Majority Position", progress.majority_position)
        metrics_table.add_row("LLM Calls", str(progress.llm_calls))
        metrics_table.add_row("LLM Cost", f"${progress.llm_cost_usd:.4f}")

        # Positions table
        if progress.positions:
            positions_table = Table(title="Agent Positions", box=box.ROUNDED)
            positions_table.add_column("Agent", style="cyan")
            positions_table.add_column("Position", style="green")
            positions_table.add_column("Bar", style="yellow")

            for agent_id, position in sorted(progress.positions.items()):
                bar_len = int(position * 20)
                bar = "▓" * bar_len + "░" * (20 - bar_len)

                if position > 0.6:
                    stance = "[green]FOR[/]"
                elif position < 0.4:
                    stance = "[red]AGAINST[/]"
                else:
                    stance = "[yellow]UNDECIDED[/]"

                positions_table.add_row(agent_id, f"{position:.2f} {stance}", bar)

            self.console.print(positions_table)

        self.console.print(metrics_table)

        # Current activity
        if progress.current_speaker:
            self.console.print(
                Panel(
                    f"[bold]{progress.current_speaker}[/] ({progress.current_phase})\n\n"
                    f"{progress.last_message[:200]}..." if len(progress.last_message) > 200 else progress.last_message,
                    title="Current Speaker",
                    border_style="green"
                )
            )

        # Errors
        if progress.errors:
            for error in progress.errors[-3:]:  # Show last 3 errors
                self.console.print(f"[red]Error: {error}[/]")

    async def run_live(self, reporter: ProgressReporter, experiment_id: str):
        """Run live updating display."""
        self._running = True

        if self.rich_available:
            from rich.live import Live

            with Live(console=self.console, refresh_per_second=1/self.refresh_rate) as live:
                while self._running:
                    progress = reporter.get_progress(experiment_id)
                    if progress:
                        self._display_rich(progress)
                    await asyncio.sleep(self.refresh_rate)
        else:
            while self._running:
                progress = reporter.get_progress(experiment_id)
                if progress:
                    self._display_simple(progress)
                await asyncio.sleep(self.refresh_rate)

    def stop(self):
        """Stop the live display."""
        self._running = False


class BatchProgressUI:
    """UI for tracking multiple experiments in a batch."""

    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter

        try:
            from rich.console import Console
            from rich.table import Table
            self.console = Console()
            self.rich_available = True
        except ImportError:
            self.rich_available = False

    def display(self):
        """Display batch progress summary."""
        experiments = self.reporter.get_all_progress()

        if not experiments:
            print("No experiments running")
            return

        if self.rich_available:
            from rich.table import Table
            from rich import box

            table = Table(title="Batch Progress", box=box.ROUNDED)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Status", style="green")
            table.add_column("Progress", style="yellow")
            table.add_column("Messages", style="blue")
            table.add_column("Consensus", style="magenta")

            for exp_id, progress in experiments.items():
                bar_len = int(progress.percent_complete / 100 * 10)
                bar = "▓" * bar_len + "░" * (10 - bar_len)

                table.add_row(
                    progress.experiment_id[:8],
                    progress.name[:20],
                    progress.status,
                    f"{bar} {progress.percent_complete:.0f}%",
                    str(progress.total_messages),
                    f"{progress.consensus_score:.2f}"
                )

            self.console.clear()
            self.console.print(table)
        else:
            print("\n--- Batch Progress ---")
            for exp_id, progress in experiments.items():
                print(
                    f"{progress.experiment_id[:8]} | {progress.name[:20]} | "
                    f"{progress.status} | {progress.percent_complete:.0f}% | "
                    f"Messages: {progress.total_messages}"
                )


def watch_progress_file(filepath: str, callback: Callable[[Dict], None]):
    """Watch a progress file for updates and call callback on changes."""
    path = Path(filepath)
    last_mtime = 0

    while True:
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
                try:
                    with open(path) as f:
                        data = json.load(f)
                    callback(data)
                except Exception as e:
                    logger.warning(f"Error reading progress file: {e}")

        time.sleep(0.5)


if __name__ == "__main__":
    # Demo mode
    import random

    async def demo():
        progress = ExperimentProgress(
            experiment_id="demo123",
            name="Demo Experiment",
            status="running",
            total_rounds=10,
            agents_total=6,
            positions={
                "sam_altman": 0.7,
                "yann_lecun": 0.3,
                "yoshua_bengio": 0.6,
                "elon_musk": 0.4,
                "geoffrey_hinton": 0.5,
                "dario_amodei": 0.8,
            }
        )

        ui = TerminalUI()

        for i in range(1, 11):
            progress.current_round = i
            progress.total_messages += random.randint(3, 8)
            progress.consensus_score = random.uniform(0.3, 0.8)
            progress.elapsed_seconds = i * 12
            progress.current_speaker = random.choice(list(progress.positions.keys()))
            progress.last_message = f"This is a sample message from round {i}..."

            # Random position drift
            for agent in progress.positions:
                progress.positions[agent] += random.uniform(-0.05, 0.05)
                progress.positions[agent] = max(0, min(1, progress.positions[agent]))

            progress.update_percent()
            progress.estimate_remaining()

            ui.display_progress(progress)
            await asyncio.sleep(1)

        progress.status = "completed"
        ui.display_progress(progress)

    asyncio.run(demo())
