"""Live dashboard for the multi-agent system."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from .theme import Theme, Color, SYMBOLS


class Dashboard:
    """
    Live-updating dashboard for monitoring the multi-agent system.
    Uses Rich's Live display for smooth updates.
    """

    def __init__(self, theme: Optional[Theme] = None, refresh_rate: float = 4.0):
        self.theme = theme or Theme.dark()
        self.refresh_rate = refresh_rate
        self._console = Console(force_terminal=True, color_system="truecolor")
        self._start_time = datetime.now(timezone.utc)
        self._running = False

        # State
        self._agents: Dict[str, dict] = {}
        self._messages: List[dict] = []
        self._events: List[dict] = []
        self._stats: Dict[str, Any] = {}
        self._max_log_lines = 15
        self._max_message_lines = 10

    def _elapsed(self) -> str:
        elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        hours, rem = divmod(int(elapsed), 3600)
        mins, secs = divmod(rem, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    # ═══════════════════════════════════════════════════════════════════════════
    # STATE UPDATES
    # ═══════════════════════════════════════════════════════════════════════════

    def update_agent(self, agent_id: str, data: dict):
        """Update agent state."""
        self._agents[agent_id] = {
            **self._agents.get(agent_id, {}),
            **data,
            "last_update": datetime.now(timezone.utc),
        }

    def remove_agent(self, agent_id: str):
        """Remove agent from dashboard."""
        self._agents.pop(agent_id, None)

    def add_message(self, msg: dict):
        """Add message to log."""
        self._messages.append({
            **msg,
            "timestamp": datetime.now(timezone.utc),
        })
        # Keep only recent messages
        if len(self._messages) > self._max_message_lines * 2:
            self._messages = self._messages[-self._max_message_lines:]

    def add_event(self, event: dict):
        """Add event to log."""
        self._events.append({
            **event,
            "timestamp": datetime.now(timezone.utc),
        })
        # Keep only recent events
        if len(self._events) > self._max_log_lines * 2:
            self._events = self._events[-self._max_log_lines:]

    def update_stats(self, stats: dict):
        """Update system statistics."""
        self._stats.update(stats)

    # ═══════════════════════════════════════════════════════════════════════════
    # LAYOUT COMPONENTS
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_header(self) -> Panel:
        """Create header panel."""
        title = Text()
        title.append("🔮 ", style="bold")
        title.append("LIDA", style=f"bold {Color.PRIMARY.value}")
        title.append(" Multi-Agent System", style=f"bold {Color.SECONDARY.value}")

        subtitle = Text()
        subtitle.append(f"Uptime: {self._elapsed()}", style=Color.MUTED.value)
        subtitle.append("  │  ", style=Color.BORDER.value)
        subtitle.append(f"Agents: {len(self._agents)}", style=Color.INFO.value)
        subtitle.append("  │  ", style=Color.BORDER.value)

        active = sum(1 for a in self._agents.values() if a.get("status") in ("running", "ready", "busy"))
        subtitle.append(f"Active: {active}", style=Color.SUCCESS.value)

        content = Group(title, subtitle)

        return Panel(
            content,
            border_style=Color.BORDER.value,
            box=box.DOUBLE,
            padding=(0, 2),
        )

    def _make_agents_panel(self) -> Panel:
        """Create agents status panel."""
        table = Table(
            box=box.SIMPLE,
            show_header=True,
            header_style=f"bold {Color.SECONDARY.value}",
            border_style=Color.BORDER.value,
            expand=True,
            padding=(0, 1),
        )

        table.add_column("Agent", style="bold", no_wrap=True)
        table.add_column("Type", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("In", justify="right", no_wrap=True)
        table.add_column("Out", justify="right", no_wrap=True)
        table.add_column("Proc", justify="right", no_wrap=True)

        for agent_id, data in sorted(self._agents.items()):
            status = data.get("status", "unknown")
            agent_type = data.get("type", "unknown")
            status_color = self.theme.get_status_color(status)
            agent_color = self.theme.get_agent_color(agent_type)
            symbol = SYMBOLS.get(agent_type.lower(), SYMBOLS["dot"])

            table.add_row(
                f"{agent_id[:20]}",
                f"[{agent_color}]{symbol} {agent_type}[/]",
                f"[{status_color}]{status}[/]",
                str(data.get("inbox", 0)),
                str(data.get("outbox", 0)),
                str(data.get("processed", 0)),
            )

        return Panel(
            table,
            title="[bold]Agents[/]",
            border_style=Color.BORDER.value,
            box=box.ROUNDED,
        )

    def _make_messages_panel(self) -> Panel:
        """Create messages panel."""
        lines = []

        for msg in self._messages[-self._max_message_lines:]:
            ts = msg.get("timestamp", datetime.now(timezone.utc))
            elapsed = (ts - self._start_time).total_seconds()

            msg_type = msg.get("type", "direct")
            sender = msg.get("sender", "?")[:12]
            recipient = msg.get("recipient", "?")[:12]

            color = self.theme.get_message_color(msg_type)
            symbol = SYMBOLS.get(msg_type.lower(), SYMBOLS["direct"])

            if msg_type.lower() == "broadcast":
                arrow = SYMBOLS["broadcast"]
                recipient_str = f"[{Color.BROADCAST.value}]ALL[/]"
            elif msg_type.lower() == "multicast":
                arrow = SYMBOLS["multicast"]
                recipient_str = f"[{Color.MULTICAST.value}]{recipient}[/]"
            else:
                arrow = SYMBOLS["arrow_right"]
                recipient_str = recipient

            line = Text()
            line.append(f"{elapsed:7.2f}s ", style=Color.MUTED.value)
            line.append(f"{symbol} ", style=color)
            line.append(f"{sender} ", style="bold")
            line.append(f"{arrow} ", style=Color.MUTED.value)
            line.append(recipient_str)

            lines.append(line)

        if not lines:
            lines.append(Text("No messages yet...", style=Color.MUTED.value))

        return Panel(
            Group(*lines),
            title="[bold]Messages[/]",
            border_style=Color.BORDER.value,
            box=box.ROUNDED,
        )

    def _make_events_panel(self) -> Panel:
        """Create events log panel."""
        lines = []

        for event in self._events[-self._max_log_lines:]:
            ts = event.get("timestamp", datetime.now(timezone.utc))
            elapsed = (ts - self._start_time).total_seconds()

            level = event.get("level", "info")
            source = event.get("source", "")[:15]
            message = event.get("message", "")[:60]

            colors = {
                "debug": Color.MUTED.value,
                "info": Color.INFO.value,
                "success": Color.SUCCESS.value,
                "warning": Color.WARNING.value,
                "error": Color.ERROR.value,
            }
            symbols = {
                "debug": SYMBOLS["dot"],
                "info": SYMBOLS["info"],
                "success": SYMBOLS["check"],
                "warning": SYMBOLS["warning"],
                "error": SYMBOLS["cross"],
            }

            color = colors.get(level, Color.MUTED.value)
            symbol = symbols.get(level, SYMBOLS["dot"])

            line = Text()
            line.append(f"{elapsed:7.2f}s ", style=Color.MUTED.value)
            line.append(f"{symbol} ", style=color)
            if source:
                line.append(f"{source:>15} ", style=Color.MUTED.value)
            line.append(message)

            lines.append(line)

        if not lines:
            lines.append(Text("No events yet...", style=Color.MUTED.value))

        return Panel(
            Group(*lines),
            title="[bold]Events[/]",
            border_style=Color.BORDER.value,
            box=box.ROUNDED,
        )

    def _make_stats_panel(self) -> Panel:
        """Create statistics panel."""
        table = Table(
            box=None,
            show_header=False,
            padding=(0, 2),
            expand=True,
        )

        table.add_column("Metric", style=Color.MUTED.value)
        table.add_column("Value", justify="right", style="bold")

        stats_items = [
            ("Messages Sent", self._stats.get("messages_sent", 0)),
            ("Messages Recv", self._stats.get("messages_received", 0)),
            ("Broadcasts", self._stats.get("broadcasts", 0)),
            ("Multicasts", self._stats.get("multicasts", 0)),
            ("Errors", self._stats.get("errors", 0)),
            ("Avg Latency", f"{self._stats.get('avg_latency_ms', 0):.1f}ms"),
        ]

        for metric, value in stats_items:
            table.add_row(metric, str(value))

        return Panel(
            table,
            title="[bold]Statistics[/]",
            border_style=Color.BORDER.value,
            box=box.ROUNDED,
        )

    def _make_layout(self) -> Layout:
        """Create the full dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )

        layout["left"].split_column(
            Layout(name="agents"),
            Layout(name="stats", size=12),
        )

        layout["right"].split_column(
            Layout(name="messages"),
            Layout(name="events"),
        )

        # Populate
        layout["header"].update(self._make_header())
        layout["agents"].update(self._make_agents_panel())
        layout["stats"].update(self._make_stats_panel())
        layout["messages"].update(self._make_messages_panel())
        layout["events"].update(self._make_events_panel())

        # Footer
        footer = Text()
        footer.append(" Press ", style=Color.MUTED.value)
        footer.append("Ctrl+C", style=f"bold {Color.WARNING.value}")
        footer.append(" to stop ", style=Color.MUTED.value)
        footer.append("│", style=Color.BORDER.value)
        footer.append(f" {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC ", style=Color.MUTED.value)

        layout["footer"].update(Panel(footer, box=box.SIMPLE, border_style=Color.BORDER.value))

        return layout

    # ═══════════════════════════════════════════════════════════════════════════
    # RUNNING
    # ═══════════════════════════════════════════════════════════════════════════

    async def run(self, update_fn: Optional[Callable] = None):
        """Run the dashboard with live updates."""
        self._running = True

        with Live(
            self._make_layout(),
            console=self._console,
            refresh_per_second=self.refresh_rate,
            screen=True,
        ) as live:
            try:
                while self._running:
                    if update_fn:
                        await update_fn(self)
                    live.update(self._make_layout())
                    await asyncio.sleep(1 / self.refresh_rate)
            except KeyboardInterrupt:
                self._running = False

    def stop(self):
        """Stop the dashboard."""
        self._running = False

    def render_once(self):
        """Render the dashboard once (for testing)."""
        self._console.print(self._make_layout())
