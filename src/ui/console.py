"""Rich console output for the multi-agent system."""
from __future__ import annotations

import sys
from datetime import datetime
from typing import Any, Optional, List

from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich.style import Style
from rich import box

from .theme import Theme, Color, SYMBOLS


class Console:
    """
    Styled console output for the multi-agent system.
    Provides beautiful, consistent formatting for all output.
    """

    def __init__(self, theme: Optional[Theme] = None):
        self.theme = theme or Theme.dark()
        self._console = RichConsole(
            force_terminal=True,
            color_system="truecolor",
        )
        self._start_time = datetime.utcnow()

    @property
    def width(self) -> int:
        return self._console.width

    def _timestamp(self) -> str:
        elapsed = (datetime.utcnow() - self._start_time).total_seconds()
        return f"[{Color.MUTED.value}]{elapsed:8.2f}s[/]"

    # ═══════════════════════════════════════════════════════════════════════════
    # BASIC OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════

    def print(self, *args, **kwargs):
        """Print to console."""
        self._console.print(*args, **kwargs)

    def log(self, message: str, level: str = "info", source: Optional[str] = None):
        """Log a message with timestamp and level."""
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

        source_str = f"[{Color.MUTED.value}]{source:>15}[/] " if source else ""
        self._console.print(
            f"{self._timestamp()} [{color}]{symbol}[/] {source_str}{message}"
        )

    def debug(self, message: str, source: Optional[str] = None):
        self.log(message, "debug", source)

    def info(self, message: str, source: Optional[str] = None):
        self.log(message, "info", source)

    def success(self, message: str, source: Optional[str] = None):
        self.log(message, "success", source)

    def warning(self, message: str, source: Optional[str] = None):
        self.log(message, "warning", source)

    def error(self, message: str, source: Optional[str] = None):
        self.log(message, "error", source)

    # ═══════════════════════════════════════════════════════════════════════════
    # HEADERS & SECTIONS
    # ═══════════════════════════════════════════════════════════════════════════

    def header(self, title: str, subtitle: Optional[str] = None):
        """Print a styled header."""
        self._console.print()
        self._console.print(
            Panel(
                Text(title, style=f"bold {Color.PRIMARY.value}", justify="center"),
                subtitle=subtitle,
                border_style=Color.BORDER.value,
                box=box.DOUBLE,
                padding=(1, 2),
            )
        )
        self._console.print()

    def section(self, title: str):
        """Print a section divider."""
        self._console.print()
        self._console.print(
            f"[{Color.BORDER.value}]{'─' * 3}[/] "
            f"[bold {Color.SECONDARY.value}]{title}[/] "
            f"[{Color.BORDER.value}]{'─' * (self.width - len(title) - 6)}[/]"
        )
        self._console.print()

    def divider(self, char: str = "─"):
        """Print a divider line."""
        self._console.print(f"[{Color.BORDER.value}]{char * self.width}[/]")

    # ═══════════════════════════════════════════════════════════════════════════
    # AGENT OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════

    def agent_spawned(self, agent_id: str, agent_type: str):
        """Log agent spawn event."""
        color = self.theme.get_agent_color(agent_type)
        symbol = SYMBOLS.get(agent_type.lower(), SYMBOLS["dot"])
        self._console.print(
            f"{self._timestamp()} [{Color.SUCCESS.value}]{SYMBOLS['check']}[/] "
            f"[{color}]{symbol} {agent_type.upper()}[/] "
            f"[bold]{agent_id}[/] spawned"
        )

    def agent_terminated(self, agent_id: str, agent_type: str):
        """Log agent termination."""
        color = self.theme.get_agent_color(agent_type)
        symbol = SYMBOLS.get(agent_type.lower(), SYMBOLS["dot"])
        self._console.print(
            f"{self._timestamp()} [{Color.MUTED.value}]{SYMBOLS['cross']}[/] "
            f"[{color}]{symbol} {agent_type.upper()}[/] "
            f"[bold]{agent_id}[/] terminated"
        )

    def agent_status(self, agent_id: str, status: str, details: Optional[str] = None):
        """Log agent status change."""
        color = self.theme.get_status_color(status)
        details_str = f" [{Color.MUTED.value}]({details})[/]" if details else ""
        self._console.print(
            f"{self._timestamp()} [{color}]{SYMBOLS['dot']}[/] "
            f"[bold]{agent_id}[/] → [{color}]{status.upper()}[/]{details_str}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # MESSAGE OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════

    def message(
        self,
        msg_type: str,
        sender: str,
        recipient: str,
        summary: Optional[str] = None,
    ):
        """Log a message event."""
        color = self.theme.get_message_color(msg_type)
        symbol = SYMBOLS.get(msg_type.lower(), SYMBOLS["direct"])

        if msg_type.lower() == "broadcast":
            recipient_str = f"[{Color.BROADCAST.value}]ALL[/]"
            arrow = SYMBOLS["broadcast"]
        elif msg_type.lower() == "multicast":
            recipient_str = f"[{Color.MULTICAST.value}]{recipient}[/]"
            arrow = SYMBOLS["multicast"]
        else:
            recipient_str = f"[bold]{recipient}[/]"
            arrow = SYMBOLS["arrow_right"]

        summary_str = f" [{Color.MUTED.value}]│ {summary[:40]}...[/]" if summary and len(summary) > 40 else (f" [{Color.MUTED.value}]│ {summary}[/]" if summary else "")

        self._console.print(
            f"{self._timestamp()} [{color}]{symbol}[/] "
            f"[bold]{sender}[/] {arrow} {recipient_str}"
            f"{summary_str}"
        )

    def message_flow(self, sender: str, recipient: str, msg_type: str, direction: str = "out"):
        """Show message flow with animation-friendly format."""
        color = self.theme.get_message_color(msg_type)
        if direction == "out":
            self._console.print(
                f"  [{Color.MUTED.value}]├──[/] [{color}]{msg_type.upper():12}[/] "
                f"[bold]{sender}[/] {SYMBOLS['arrow_right']} [bold]{recipient}[/]"
            )
        else:
            self._console.print(
                f"  [{Color.MUTED.value}]├──[/] [{color}]{msg_type.upper():12}[/] "
                f"[bold]{recipient}[/] {SYMBOLS['arrow_left']} [bold]{sender}[/]"
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # TABLES
    # ═══════════════════════════════════════════════════════════════════════════

    def agents_table(self, agents: List[dict]):
        """Display a table of agents."""
        table = Table(
            title="[bold]Agents[/]",
            box=box.ROUNDED,
            border_style=Color.BORDER.value,
            header_style=f"bold {Color.SECONDARY.value}",
            row_styles=["", f"dim"],
        )

        table.add_column("ID", style="bold")
        table.add_column("Type")
        table.add_column("Status")
        table.add_column("Inbox")
        table.add_column("Outbox")
        table.add_column("Processed")

        for agent in agents:
            status = agent.get("status", "unknown")
            status_color = self.theme.get_status_color(status)
            agent_color = self.theme.get_agent_color(agent.get("type", ""))

            table.add_row(
                agent.get("id", "?"),
                f"[{agent_color}]{agent.get('type', '?')}[/]",
                f"[{status_color}]{status}[/]",
                str(agent.get("inbox", 0)),
                str(agent.get("outbox", 0)),
                str(agent.get("processed", 0)),
            )

        self._console.print(table)

    def stats_table(self, stats: dict):
        """Display statistics table."""
        table = Table(
            title="[bold]Statistics[/]",
            box=box.ROUNDED,
            border_style=Color.BORDER.value,
            show_header=False,
        )

        table.add_column("Metric", style=f"bold {Color.MUTED.value}")
        table.add_column("Value", justify="right")

        for key, value in stats.items():
            table.add_row(key, str(value))

        self._console.print(table)

    # ═══════════════════════════════════════════════════════════════════════════
    # SPECIAL DISPLAYS
    # ═══════════════════════════════════════════════════════════════════════════

    def topology(self, nodes: List[str], edges: List[tuple]):
        """Display a simple network topology."""
        self._console.print()
        self._console.print(f"[bold {Color.SECONDARY.value}]Network Topology[/]")
        self._console.print()

        # Simple text-based representation
        for node in nodes:
            connections = [e[1] for e in edges if e[0] == node]
            if connections:
                conn_str = ", ".join(connections[:3])
                if len(connections) > 3:
                    conn_str += f" +{len(connections)-3} more"
                self._console.print(
                    f"  [{Color.SECONDARY.value}]{SYMBOLS['dot']}[/] "
                    f"[bold]{node}[/] {SYMBOLS['arrow_right']} {conn_str}"
                )
            else:
                self._console.print(
                    f"  [{Color.MUTED.value}]{SYMBOLS['ring']}[/] "
                    f"[bold]{node}[/]"
                )

    def progress_bar(self, current: int, total: int, width: int = 40, label: str = ""):
        """Display a progress bar."""
        filled = int(width * current / total) if total > 0 else 0
        bar = SYMBOLS["bar_full"] * filled + SYMBOLS["bar_light"] * (width - filled)
        percent = (current / total * 100) if total > 0 else 0
        self._console.print(
            f"  {label:20} [{Color.SECONDARY.value}]{bar}[/] "
            f"[{Color.MUTED.value}]{percent:5.1f}%[/]"
        )


# Global console instance
console = Console()
