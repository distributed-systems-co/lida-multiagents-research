#!/usr/bin/env python3
"""
Live Multi-Agent Swarm Dashboard

A visually impressive real-time dashboard showing:
- Swarm of agents with distinct personalities
- Message flow visualization
- Deliberation and consensus building
- Network topology
- Real-time statistics
"""

import asyncio
import os
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.columns import Columns
from rich import box
from rich.style import Style
from rich.align import Align

# Import from your codebase
from src.meta.personality import (
    get_personality_manager,
    PERSONALITY_ARCHETYPES,
    TraitDimension,
)

console = Console(force_terminal=True, color_system="truecolor")

# ═══════════════════════════════════════════════════════════════════════════════
# COLORS & SYMBOLS
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_COLORS = [
    "#FF6B6B",  # Coral
    "#4ECDC4",  # Teal
    "#45B7D1",  # Sky Blue
    "#96CEB4",  # Sage
    "#FFEAA7",  # Pale Yellow
    "#DDA0DD",  # Plum
    "#98D8C8",  # Mint
    "#F7DC6F",  # Golden
    "#BB8FCE",  # Lavender
    "#85C1E9",  # Light Blue
    "#F8B500",  # Amber
    "#00CED1",  # Dark Cyan
]

AGENT_SYMBOLS = ["◆", "●", "▲", "■", "★", "◈", "◉", "▼", "◐", "◑", "⬟", "⬡"]

PERSONALITY_EMOJIS = {
    "the_scholar": "📚",
    "the_pragmatist": "🎯",
    "the_creative": "🎨",
    "the_skeptic": "🔍",
    "the_mentor": "🎓",
    "the_synthesizer": "🔮",
}

MESSAGE_TYPES = {
    "broadcast": ("📡", "#EC4899"),
    "direct": ("✉️", "#06B6D4"),
    "propose": ("💡", "#F59E0B"),
    "vote": ("🗳️", "#8B5CF6"),
    "agree": ("✓", "#10B981"),
    "disagree": ("✗", "#EF4444"),
    "think": ("💭", "#6B7280"),
    "respond": ("💬", "#3B82F6"),
}

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BeliefState:
    """Tracks an agent's belief on a topic."""
    topic: str
    position: float  # -1.0 (oppose) to 1.0 (support)
    confidence: float  # 0.0 to 1.0
    timestamp: float

@dataclass
class SwarmAgent:
    """Represents an agent in the swarm."""
    id: str
    name: str
    personality: str
    color: str
    symbol: str
    emoji: str
    status: str = "idle"
    thinking: str = ""
    last_message: str = ""
    messages_sent: int = 0
    messages_received: int = 0
    votes: Dict[str, str] = field(default_factory=dict)
    position: tuple = (0, 0)  # For network visualization
    connections: List[str] = field(default_factory=list)
    activity_level: float = 0.0
    current_task: str = ""
    # Belief tracking
    belief_position: float = 0.0  # -1 to 1
    belief_confidence: float = 0.5
    belief_history: List[tuple] = field(default_factory=list)  # (timestamp, position, confidence)
    influence_received: Dict[str, float] = field(default_factory=dict)  # who influenced this agent
    current_argument: str = ""  # What argument they're making


@dataclass
class DeliberationContext:
    """Tracks the full context of what's being debated."""
    topic: str = ""
    topic_description: str = ""
    proposal_options: List[str] = field(default_factory=list)
    selection_criterion: str = ""  # e.g., "majority", "supermajority", "consensus"
    threshold: float = 0.5  # Vote threshold for approval
    phase_descriptions: Dict[str, str] = field(default_factory=lambda: {
        "initializing": "Setting up the deliberation framework",
        "opening": "Agents forming initial positions based on their expertise",
        "positions": "Each agent stating their initial stance publicly",
        "debating": "Agents exchanging arguments and updating beliefs",
        "voting": "Final votes being cast based on deliberation",
        "synthesis": "Aggregating votes and forming consensus",
        "complete": "Deliberation concluded with decision",
    })
    key_arguments: List[tuple] = field(default_factory=list)  # (agent_id, argument, for/against)


@dataclass
class SwarmMessage:
    """A message in the swarm."""
    timestamp: float
    sender: str
    recipient: str  # "all" for broadcast
    msg_type: str
    content: str


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TickerMessage:
    """A message with ticker scroll state."""
    msg: SwarmMessage
    scroll_offset: float = 0.0
    creation_frame: int = 0

class SwarmDashboard:
    """Live dashboard for multi-agent swarm visualization."""

    def __init__(self, num_agents: int = 8):
        self.num_agents = num_agents
        self.agents: Dict[str, SwarmAgent] = {}
        self.messages: deque = deque(maxlen=100)
        self.ticker_messages: List[TickerMessage] = []
        self.start_time = time.time()
        self.frame = 0
        self.topic = ""
        self.deliberation_phase = "initializing"
        self.consensus: Dict[str, int] = {}
        self.total_messages = 0
        self.active_connections: List[tuple] = []

        # Deliberation context - explains WHAT is being debated and HOW
        self.context = DeliberationContext()

        # Ticker lane configuration: (speed_multiplier, max_messages)
        self.ticker_lanes = [
            {"speed": 0.0, "label": "LIVE"},      # Lane 0: newest, static
            {"speed": 0.3, "label": "RECENT"},    # Lane 1: slow scroll
            {"speed": 0.8, "label": "HISTORY"},   # Lane 2: medium scroll
            {"speed": 1.5, "label": "ARCHIVE"},   # Lane 3: fast scroll
        ]

        # Initialize personality manager
        self.pm = get_personality_manager()

        # Initialize agents
        self._init_agents()

    def _init_agents(self):
        """Initialize swarm agents with different personalities."""
        archetypes = list(PERSONALITY_ARCHETYPES.keys())

        for i in range(self.num_agents):
            agent_id = f"agent-{i:02d}"
            archetype = archetypes[i % len(archetypes)]

            # Create personality
            personality = self.pm.create(
                name=f"swarm-{agent_id}",
                archetype=archetype
            )

            agent = SwarmAgent(
                id=agent_id,
                name=personality.name,
                personality=archetype,
                color=AGENT_COLORS[i % len(AGENT_COLORS)],
                symbol=AGENT_SYMBOLS[i % len(AGENT_SYMBOLS)],
                emoji=PERSONALITY_EMOJIS.get(archetype, "🤖"),
                position=self._get_ring_position(i, self.num_agents),
            )

            # Connect to neighbors
            for j in range(self.num_agents):
                if i != j:
                    agent.connections.append(f"agent-{j:02d}")

            self.agents[agent_id] = agent

    def _get_ring_position(self, index: int, total: int) -> tuple:
        """Get position on a ring layout."""
        import math
        angle = (2 * math.pi * index) / total - math.pi / 2
        radius = 8
        x = int(radius * math.cos(angle))
        y = int(radius * math.sin(angle) * 0.5)  # Compress vertically
        return (x + 12, y + 6)  # Center offset

    def elapsed(self) -> str:
        """Get elapsed time string."""
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        return f"{mins:02d}:{secs:02d}"

    def add_message(self, sender: str, recipient: str, msg_type: str, content: str):
        """Add a message to the swarm."""
        msg = SwarmMessage(
            timestamp=time.time() - self.start_time,
            sender=sender,
            recipient=recipient,
            msg_type=msg_type,
            content=content,  # No truncation - store full content
        )
        self.messages.append(msg)
        self.total_messages += 1

        # Add to ticker with current frame
        ticker_msg = TickerMessage(msg=msg, scroll_offset=0.0, creation_frame=self.frame)
        self.ticker_messages.append(ticker_msg)

        # Keep ticker messages bounded
        if len(self.ticker_messages) > 60:
            self.ticker_messages = self.ticker_messages[-60:]

        # Update agent stats
        if sender in self.agents:
            self.agents[sender].messages_sent += 1
            self.agents[sender].last_message = content
            self.agents[sender].activity_level = min(1.0, self.agents[sender].activity_level + 0.3)

        if recipient != "all" and recipient in self.agents:
            self.agents[recipient].messages_received += 1
        elif recipient == "all":
            for agent in self.agents.values():
                if agent.id != sender:
                    agent.messages_received += 1

    def set_agent_status(self, agent_id: str, status: str, thinking: str = ""):
        """Update agent status."""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].thinking = thinking

    def set_agent_vote(self, agent_id: str, proposal: str, vote: str):
        """Record agent vote."""
        if agent_id in self.agents:
            self.agents[agent_id].votes[proposal] = vote
            if vote not in self.consensus:
                self.consensus[vote] = 0
            self.consensus[vote] += 1

    def update_agent_belief(self, agent_id: str, position: float, confidence: float, influencer: str = None):
        """Update an agent's belief state."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            elapsed = time.time() - self.start_time
            agent.belief_position = max(-1.0, min(1.0, position))
            agent.belief_confidence = max(0.0, min(1.0, confidence))
            agent.belief_history.append((elapsed, position, confidence))
            # Keep last 20 data points
            if len(agent.belief_history) > 20:
                agent.belief_history.pop(0)
            if influencer:
                agent.influence_received[influencer] = agent.influence_received.get(influencer, 0) + 1

    # ═══════════════════════════════════════════════════════════════════════════
    # RENDERING
    # ═══════════════════════════════════════════════════════════════════════════

    def _make_header(self) -> Text:
        """Create compact single-line header."""
        spinner = SPINNER_FRAMES[self.frame % len(SPINNER_FRAMES)]

        # Calculate belief stats
        avg_belief = sum(a.belief_position for a in self.agents.values()) / len(self.agents) if self.agents else 0
        avg_conf = sum(a.belief_confidence for a in self.agents.values()) / len(self.agents) if self.agents else 0

        header = Text()
        header.append(f" {spinner} ", style="bold cyan")
        header.append("LIDA", style="bold #7C3AED")
        header.append(" Swarm ", style="bold white")
        header.append("│", style="dim")
        header.append(f" ⏱{self.elapsed()}", style="#06B6D4")
        header.append(f" 📨{self.total_messages}", style="#10B981")
        active = sum(1 for a in self.agents.values() if a.status not in ("idle", "done"))
        header.append(f" ⚡{active}", style="#F59E0B")
        header.append(" │", style="dim")
        header.append(f" μ:", style="dim")
        belief_color = "#10B981" if avg_belief > 0.1 else "#EF4444" if avg_belief < -0.1 else "#F59E0B"
        header.append(f"{avg_belief:+.2f}", style=belief_color)
        header.append(f" σ:{avg_conf:.0%}", style="dim")
        header.append(" │ ", style="dim")
        phase_color = {"initializing": "dim", "opening": "#06B6D4", "positions": "#F59E0B", "debating": "#10B981", "voting": "#8B5CF6", "synthesis": "#EC4899", "complete": "#10B981"}.get(self.deliberation_phase, "white")
        header.append(f"{self.deliberation_phase.upper()}", style=f"bold {phase_color}")

        return header

    def _make_network_view(self) -> Panel:
        """Create the network topology visualization."""
        # Create a grid for the network
        width, height = 25, 13
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Draw connections (active ones highlighted)
        for agent in self.agents.values():
            ax, ay = agent.position
            if 0 <= ax < width and 0 <= ay < height:
                # Draw agent symbol
                pass  # We'll overlay agents last

        # Draw agents on the grid
        lines = []
        for y in range(height):
            line = Text()
            for x in range(width):
                placed = False
                for agent in self.agents.values():
                    ax, ay = agent.position
                    if ax == x and ay == y:
                        style = agent.color
                        if agent.status == "thinking":
                            style = f"bold {agent.color} on #1a1a2e"
                        elif agent.status == "speaking":
                            style = f"bold {agent.color} reverse"
                        line.append(agent.symbol, style=style)
                        placed = True
                        break
                if not placed:
                    line.append(" ")
            lines.append(line)

        # Add activity indicators around active agents
        content = Group(*lines)

        return Panel(
            content,
            title="[bold]🌐 Swarm Network[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    def _make_sparkline(self, history: List[tuple], width: int = 10) -> str:
        """Create a mini sparkline from belief history."""
        if not history:
            return "·" * width

        # Sparkline chars: going from -1 to 1
        spark_chars = "▁▂▃▄▅▆▇█"

        # Take last `width` points
        points = [h[1] for h in history[-width:]]

        # Normalize -1 to 1 -> 0 to 7
        line = ""
        for p in points:
            idx = int((p + 1) / 2 * 7)
            idx = max(0, min(7, idx))
            line += spark_chars[idx]

        # Pad if needed
        line = line.rjust(width, "·")
        return line

    def _make_belief_bar(self, position: float, confidence: float, width: int = 16) -> Text:
        """Create a visual belief bar: oppose ◀━━━│━━━▶ support"""
        bar = Text()
        mid = width // 2

        # Position from -1 to 1 maps to 0 to width
        pos_idx = int((position + 1) / 2 * (width - 1))

        for i in range(width):
            if i == mid:
                bar.append("│", style="dim")
            elif i == pos_idx:
                # Marker with confidence-based intensity
                color = "#EF4444" if position < 0 else "#10B981" if position > 0 else "#F59E0B"
                intensity = "bold" if confidence > 0.7 else "" if confidence > 0.4 else "dim"
                bar.append("●", style=f"{intensity} {color}")
            elif (i < pos_idx and i >= mid) or (i > pos_idx and i <= mid):
                bar.append("─", style="dim")
            else:
                bar.append("─", style="dim white")

        return bar

    def _make_agents_panel(self) -> Panel:
        """Create compact belief-focused agent panel with sparklines."""
        lines = []

        # Sort by current belief position for visual grouping
        sorted_agents = sorted(self.agents.values(), key=lambda a: -a.belief_position)

        for agent in sorted_agents:
            line = Text()

            # Status indicator (animated)
            status_chars = {
                "idle": "○",
                "thinking": SPINNER_FRAMES[self.frame % len(SPINNER_FRAMES)],
                "speaking": "◉",
                "voting": "◈",
                "done": "●",
            }
            status_char = status_chars.get(agent.status, "○")
            status_color = {
                "idle": "dim",
                "thinking": "#F59E0B",
                "speaking": "#10B981",
                "voting": "#8B5CF6",
                "done": "#6B7280",
            }.get(agent.status, "dim")

            # Agent ID (compact)
            line.append(f"{status_char}", style=status_color)
            line.append(f"{agent.symbol}", style=f"bold {agent.color}")
            line.append(f"{agent.id[-2:]} ", style=agent.color)

            # Belief bar
            line.append_text(self._make_belief_bar(agent.belief_position, agent.belief_confidence, 12))
            line.append(" ", style="")

            # Sparkline history
            spark = self._make_sparkline(agent.belief_history, 8)
            trend_color = "#10B981" if len(agent.belief_history) > 1 and agent.belief_history[-1][1] > agent.belief_history[-2][1] else "#EF4444" if len(agent.belief_history) > 1 and agent.belief_history[-1][1] < agent.belief_history[-2][1] else "#6B7280"
            line.append(spark, style=trend_color)

            # Confidence %
            conf_pct = int(agent.belief_confidence * 100)
            line.append(f" {conf_pct:2d}%", style="dim")

            lines.append(line)

        return Panel(
            Group(*lines),
            title="[bold]◀OPPOSE│SUPPORT▶  History  Conf[/]",
            title_align="left",
            border_style="#374151",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    def _format_message_full(self, msg: SwarmMessage, width: int) -> List[Text]:
        """Format a message with full content, wrapping to multiple lines if needed."""
        lines = []

        sender = self.agents.get(msg.sender)
        sender_sym = sender.symbol if sender else "?"
        sender_color = sender.color if sender else "white"

        if msg.recipient == "all":
            recip_str = "ALL"
            recip_style = "#EC4899"
        else:
            recipient = self.agents.get(msg.recipient)
            recip_str = recipient.symbol if recipient else "?"
            recip_style = recipient.color if recipient else "white"

        # Header line: timestamp + sender → recipient
        header = Text()
        header.append(f"{msg.timestamp:5.1f}s ", style="dim")
        header.append(f"{sender_sym}", style=f"bold {sender_color}")
        header.append("→", style="dim")
        header.append(recip_str, style=recip_style)
        header.append(" ", style="")

        # Calculate remaining width for content on first line
        header_len = 12  # approximate header length
        content_width = width - header_len

        content = msg.content
        if len(content) <= content_width:
            header.append(content, style="white")
            lines.append(header)
        else:
            # First line with header
            header.append(content[:content_width], style="white")
            lines.append(header)

            # Continuation lines (indented)
            remaining = content[content_width:]
            indent = "         "  # 9 spaces for alignment
            cont_width = width - len(indent)

            while remaining:
                cont_line = Text()
                cont_line.append(indent, style="")
                chunk = remaining[:cont_width]
                cont_line.append(chunk, style="white dim")
                lines.append(cont_line)
                remaining = remaining[cont_width:]

        return lines

    def _make_messages_panel(self) -> Panel:
        """Create ticker-style message stream with variable scroll speeds."""
        all_lines = []
        display_width = 58  # Approximate panel width

        # Update scroll offsets for all ticker messages
        for i, tm in enumerate(self.ticker_messages):
            age = self.frame - tm.creation_frame

            # Determine lane based on age (frames at 8fps)
            if age < 24:        # ~3 seconds - LIVE
                speed = 0.0
            elif age < 80:      # ~10 seconds - RECENT
                speed = 0.15    # Slow crawl
            elif age < 200:     # ~25 seconds - HISTORY
                speed = 0.4     # Medium scroll
            else:               # ARCHIVE
                speed = 0.8     # Faster scroll

            tm.scroll_offset += speed

        # Group messages by lane (matching speed thresholds)
        lanes = {0: [], 1: [], 2: [], 3: []}
        for tm in self.ticker_messages[-50:]:  # Last 50 messages
            age = self.frame - tm.creation_frame
            if age < 24:
                lanes[0].append(tm)
            elif age < 80:
                lanes[1].append(tm)
            elif age < 200:
                lanes[2].append(tm)
            else:
                lanes[3].append(tm)

        # Render each lane
        lane_labels = ["⚡LIVE", "◐RECENT", "◑HISTORY", "○ARCHIVE"]
        lane_colors = ["#10B981", "#06B6D4", "#F59E0B", "#6B7280"]

        for lane_idx in range(4):
            lane_msgs = lanes[lane_idx]
            if not lane_msgs:
                continue

            # Lane header
            label_line = Text()
            label_line.append(f"─{lane_labels[lane_idx]}─", style=f"bold {lane_colors[lane_idx]}")
            all_lines.append(label_line)

            # Render messages in this lane (newest first for LIVE/RECENT)
            if lane_idx == 0:
                lane_msgs = list(reversed(lane_msgs[-5:]))  # Show last 5, newest at top
            elif lane_idx == 1:
                lane_msgs = list(reversed(lane_msgs[-4:]))  # Show last 4
            else:
                lane_msgs = lane_msgs[-4:]  # Show last 4 for older lanes

            for tm in lane_msgs:
                msg = tm.msg
                scroll = int(tm.scroll_offset)

                # Format full message
                msg_lines = self._format_message_full(msg, display_width)

                for ml in msg_lines:
                    # Apply horizontal scroll for non-LIVE lanes
                    if lane_idx > 0 and scroll > 0:
                        plain = ml.plain
                        if scroll < len(plain):
                            # Create scrolled version
                            scrolled = Text()
                            visible = plain[scroll:scroll + display_width]
                            # Try to preserve some styling
                            scrolled.append(visible, style="dim" if lane_idx > 1 else "")
                            all_lines.append(scrolled)
                        # If scrolled past end, don't show
                    else:
                        all_lines.append(ml)

        if not all_lines:
            all_lines.append(Text("Awaiting messages...", style="dim italic"))

        # Limit total lines to fit panel (allow more to show ticker effect)
        all_lines = all_lines[:25]

        return Panel(
            Group(*all_lines),
            title="[bold]💬 Message Ticker[/]",
            border_style="#374151",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    def _make_thinking_panel(self) -> Panel:
        """Show what agents are currently thinking."""
        lines = []

        thinking_agents = [a for a in self.agents.values() if a.thinking]

        for agent in thinking_agents[:5]:
            line = Text()
            line.append(f"{agent.emoji} ", style=agent.color)
            line.append(f"{agent.id}: ", style=f"bold {agent.color}")
            line.append(agent.thinking[:50], style="italic dim")
            if len(agent.thinking) > 50:
                line.append("...", style="dim")
            lines.append(line)

        if not lines:
            spinner = SPINNER_FRAMES[(self.frame // 2) % len(SPINNER_FRAMES)]
            lines.append(Text(f"{spinner} Agents processing...", style="dim"))

        return Panel(
            Group(*lines),
            title="[bold]💭 Agent Thoughts[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    def _make_consensus_panel(self) -> Panel:
        """Show compact voting/consensus progress."""
        lines = []

        if self.consensus:
            total_votes = sum(self.consensus.values())
            for option, count in sorted(self.consensus.items(), key=lambda x: -x[1]):
                pct = (count / total_votes * 100) if total_votes > 0 else 0
                bar_len = int(pct / 10)
                bar = "█" * bar_len + "░" * (10 - bar_len)

                vote_colors = {"Support": "#10B981", "Modify": "#F59E0B", "Abstain": "#6B7280", "Oppose": "#EF4444"}
                color = vote_colors.get(option, "#6B7280")

                line = Text()
                line.append(f"{option[:8]:8}", style=f"bold {color}")
                line.append(bar, style=color)
                line.append(f"{count}", style="dim")
                lines.append(line)
        else:
            lines.append(Text("Awaiting votes...", style="dim italic"))

        return Panel(
            Group(*lines),
            title="[bold]🗳️ Votes[/]",
            border_style="#374151",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    def _make_footer(self) -> Panel:
        """Create the footer."""
        footer = Text()
        footer.append(" Press ", style="dim")
        footer.append("Ctrl+C", style="bold #F59E0B")
        footer.append(" to stop ", style="dim")
        footer.append("│", style="dim")
        footer.append(f" {datetime.now().strftime('%H:%M:%S')} ", style="dim")
        footer.append("│", style="dim")
        footer.append(" LIDA Multi-Agent Research Framework ", style="dim #7C3AED")

        return Panel(footer, box=box.SIMPLE, border_style="#374151")

    def _make_belief_heatmap(self) -> Panel:
        """Create a compact belief heatmap showing all agents."""
        lines = []

        # Row 1: belief position indicators
        pos_line = Text()
        pos_line.append("Pos:", style="dim")
        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            # Color based on belief position
            if agent.belief_position > 0.5:
                color = "#10B981"  # Strong support
            elif agent.belief_position > 0:
                color = "#34D399"  # Mild support
            elif agent.belief_position > -0.5:
                color = "#F59E0B"  # Neutral/mild oppose
            else:
                color = "#EF4444"  # Strong oppose
            pos_line.append(f" {agent.symbol}", style=f"bold {color}")
        lines.append(pos_line)

        # Row 2: confidence levels
        conf_line = Text()
        conf_line.append("Cnf:", style="dim")
        conf_chars = " ▁▂▃▄▅▆▇█"
        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            idx = int(agent.belief_confidence * 8)
            conf_line.append(f" {conf_chars[idx]}", style=agent.color)
        lines.append(conf_line)

        # Row 3: trend arrows
        trend_line = Text()
        trend_line.append("Δ  :", style="dim")
        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            if len(agent.belief_history) > 1:
                delta = agent.belief_history[-1][1] - agent.belief_history[-2][1]
                if delta > 0.1:
                    trend_line.append(" ↑", style="#10B981")
                elif delta < -0.1:
                    trend_line.append(" ↓", style="#EF4444")
                else:
                    trend_line.append(" ─", style="dim")
            else:
                trend_line.append(" ·", style="dim")
        lines.append(trend_line)

        return Panel(
            Group(*lines),
            title="[bold]Belief Matrix[/]",
            border_style="#374151",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    def make_layout(self) -> Layout:
        """Create data-dense dashboard layout with prominent deliberation context."""
        self.frame += 1

        # Decay activity levels
        for agent in self.agents.values():
            agent.activity_level = max(0, agent.activity_level - 0.02)

        layout = Layout()

        # 3-row layout with context panel prominent
        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )

        # Body: context on left (what/why), agents + messages on right (who/how)
        layout["body"].split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )

        # Left: Context panel (topic, options, criterion) + belief heatmap
        layout["left"].split_column(
            Layout(name="context"),   # PROMINENT: What are we debating? How do we decide?
            Layout(name="heatmap", size=5),   # Compact belief matrix
        )

        # Right: Agent beliefs, messages, consensus
        layout["right"].split_column(
            Layout(name="agents"),        # Agent positions with arguments
            Layout(name="messages"),      # Message stream
            Layout(name="consensus", size=6),  # Vote tallies
        )

        # Populate panels
        layout["header"].update(Panel(self._make_header(), box=box.SIMPLE, padding=0))
        layout["context"].update(self._make_context_panel())
        layout["heatmap"].update(self._make_belief_heatmap())
        layout["agents"].update(self._make_agents_panel())
        layout["messages"].update(self._make_messages_panel())
        layout["consensus"].update(self._make_consensus_panel())
        layout["footer"].update(Panel(Text(f" {datetime.now().strftime('%H:%M:%S')} │ Ctrl+C to exit │ LIDA Multi-Agent Research", style="dim"), box=box.SIMPLE, padding=0))

        return layout

    def _make_context_panel(self) -> Panel:
        """Show full deliberation context: topic, options, criterion, phase."""
        lines = []

        # ─── TOPIC ───
        if self.context.topic:
            lines.append(Text("═══ TOPIC ═══", style="bold #7C3AED"))
            # Word-wrap topic
            words = self.context.topic.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 45:
                    current_line += (" " if current_line else "") + word
                else:
                    if current_line:
                        lines.append(Text(current_line, style="bold white"))
                    current_line = word
            if current_line:
                lines.append(Text(current_line, style="bold white"))

            # Description if available
            if self.context.topic_description:
                lines.append(Text(self.context.topic_description[:60], style="dim italic"))
            lines.append(Text(""))
        else:
            lines.append(Text("Awaiting topic...", style="dim italic"))
            return Panel(Group(*lines), title="[bold]🎯 Deliberation Context[/]",
                        border_style="#374151", box=box.ROUNDED, padding=(0, 0))

        # ─── PROPOSAL OPTIONS ───
        if self.context.proposal_options:
            lines.append(Text("─── OPTIONS ───", style="bold #06B6D4"))
            vote_colors = {"Support": "#10B981", "Modify": "#F59E0B", "Abstain": "#6B7280", "Oppose": "#EF4444"}
            for opt in self.context.proposal_options:
                vote_count = self.consensus.get(opt, 0)
                total_votes = sum(self.consensus.values()) if self.consensus else 0
                pct = (vote_count / total_votes * 100) if total_votes > 0 else 0

                line = Text()
                color = vote_colors.get(opt, "#FFFFFF")
                line.append(f"  • {opt:<10}", style=color)
                if vote_count > 0:
                    bar_len = int(pct / 10)
                    bar = "█" * bar_len + "░" * (10 - bar_len)
                    line.append(f" [{bar}] ", style=color)
                    line.append(f"{vote_count} ({pct:.0f}%)", style="dim")
                lines.append(line)
            lines.append(Text(""))

        # ─── SELECTION CRITERION ───
        lines.append(Text("─── DECISION RULE ───", style="bold #F59E0B"))
        criterion_line = Text()
        criterion_line.append(f"  Method: ", style="dim")
        criterion_line.append(f"{self.context.selection_criterion}", style="bold #F59E0B")
        lines.append(criterion_line)

        threshold_line = Text()
        threshold_line.append(f"  Threshold: ", style="dim")
        threshold_line.append(f"{self.context.threshold:.0%}", style="bold")
        threshold_line.append(f" of votes", style="dim")
        lines.append(threshold_line)
        lines.append(Text(""))

        # ─── CURRENT PHASE ───
        phase_colors = {
            "initializing": "#6B7280", "opening": "#06B6D4",
            "positions": "#F59E0B", "debating": "#10B981",
            "voting": "#8B5CF6", "synthesis": "#EC4899", "complete": "#10B981"
        }
        lines.append(Text("─── PHASE ───", style=f"bold {phase_colors.get(self.deliberation_phase, '#FFFFFF')}"))
        phase_line = Text()
        phase_line.append(f"  ", style="")
        phase_line.append(f"⏵ {self.deliberation_phase.upper()}", style=f"bold {phase_colors.get(self.deliberation_phase, '#FFFFFF')}")
        lines.append(phase_line)

        # Phase description
        phase_desc = self.context.phase_descriptions.get(self.deliberation_phase, "")
        if phase_desc:
            desc_line = Text()
            desc_line.append(f"    {phase_desc[:50]}", style="dim italic")
            lines.append(desc_line)

        # ─── KEY ARGUMENTS ───
        if self.context.key_arguments:
            lines.append(Text(""))
            lines.append(Text("─── KEY ARGUMENTS ───", style="bold #EC4899"))
            for agent_id, argument, stance in self.context.key_arguments[-3:]:  # Last 3
                agent = self.agents.get(agent_id)
                arg_line = Text()
                if agent:
                    arg_line.append(f"  {agent.symbol}", style=agent.color)
                stance_color = "#10B981" if stance == "for" else "#EF4444" if stance == "against" else "#F59E0B"
                arg_line.append(f" [{stance.upper()}] ", style=stance_color)
                arg_line.append(argument[:35], style="dim")
                lines.append(arg_line)

        return Panel(
            Group(*lines),
            title="[bold]🎯 Deliberation Context[/]",
            border_style="#374151",
            box=box.ROUNDED,
            padding=(0, 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SWARM SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class SwarmSimulator:
    """Simulates swarm behavior for demonstration."""

    def __init__(self, dashboard: SwarmDashboard, use_llm: bool = False):
        self.dashboard = dashboard
        self.use_llm = use_llm
        self.llm_client = None

    async def initialize_llm(self):
        """Initialize LLM client if available."""
        if not self.use_llm:
            return

        try:
            from src.llm.openrouter import OpenRouterClient
            if os.getenv("OPENROUTER_API_KEY"):
                self.llm_client = OpenRouterClient()
                console.print("[green]✓ LLM connected[/]")
        except Exception as e:
            console.print(f"[yellow]LLM unavailable: {e}[/]")

    async def run_deliberation(self, topic: str, description: str = "", criterion: str = "majority", max_rounds: int = 5, max_voting_rounds: int = 3):
        """Run a multi-agent deliberation on a topic with belief tracking.

        Args:
            topic: The topic to deliberate on
            description: Additional context for the topic
            criterion: Voting criterion (majority, supermajority, consensus, plurality)
            max_rounds: Maximum discussion rounds
            max_voting_rounds: Maximum voting rounds before forcing decision
        """
        self.dashboard.topic = topic
        agents = list(self.dashboard.agents.values())

        # ─── SET UP DELIBERATION CONTEXT ───
        self.dashboard.context.topic = topic
        self.dashboard.context.topic_description = description
        self.dashboard.context.proposal_options = ["Support", "Modify", "Abstain", "Oppose"]
        self.dashboard.context.selection_criterion = criterion
        self.dashboard.context.threshold = {"majority": 0.5, "supermajority": 0.67, "consensus": 0.9, "plurality": 0.0}.get(criterion, 0.5)
        self.dashboard.context.key_arguments = []

        # Phase 1: Introduction - agents form initial beliefs
        self.dashboard.deliberation_phase = "opening"
        for agent in agents:
            self.dashboard.set_agent_status(agent.id, "thinking", f"Analyzing: {topic[:30]}...")
            # Initial belief based on personality
            personality_bias = {
                "the_scholar": 0.2,      # Slightly supportive, evidence-based
                "the_pragmatist": 0.0,   # Neutral, practical
                "the_creative": 0.3,     # Open to new ideas
                "the_skeptic": -0.3,     # Skeptical
                "the_mentor": 0.1,       # Cautiously supportive
                "the_synthesizer": 0.0,  # Neutral, integrative
            }
            base = personality_bias.get(agent.personality, 0)
            initial_pos = base + random.uniform(-0.4, 0.4)
            initial_conf = random.uniform(0.3, 0.6)
            self.dashboard.update_agent_belief(agent.id, initial_pos, initial_conf)
            await asyncio.sleep(0.15)

        await asyncio.sleep(0.5)

        # Phase 2: Initial positions - agents state beliefs
        self.dashboard.deliberation_phase = "positions"

        for agent in agents:
            self.dashboard.set_agent_status(agent.id, "speaking")
            pos = agent.belief_position

            if pos > 0.5:
                stance = "strongly support"
            elif pos > 0.1:
                stance = "support"
            elif pos > -0.1:
                stance = "remain neutral on"
            elif pos > -0.5:
                stance = "have concerns about"
            else:
                stance = "oppose"

            thoughts = [
                f"From a {agent.personality.replace('the_', '')} view, I {stance} this",
                f"After analysis, I {stance} the proposal",
                f"My position: {stance} ({agent.belief_confidence:.0%} confident)",
            ]

            self.dashboard.set_agent_status(agent.id, "thinking", random.choice(thoughts))
            self.dashboard.add_message(agent.id, "all", "broadcast", f"I {stance} this proposal")

            # Slight confidence increase after stating position
            self.dashboard.update_agent_belief(agent.id, pos, min(1.0, agent.belief_confidence + 0.1))
            await asyncio.sleep(0.2)

        await asyncio.sleep(0.5)

        # Phase 3: Discussion - agents influence each other with CLEAR ARGUMENTS
        self.dashboard.deliberation_phase = "debating"

        # Define substantive arguments for/against
        arguments_for = [
            "Evidence supports long-term benefits",
            "Aligns with core principles",
            "Successful precedents exist",
            "Risk-reward ratio favorable",
            "Stakeholder analysis positive",
            "Technical feasibility confirmed",
        ]
        arguments_against = [
            "Insufficient evidence of efficacy",
            "Potential unintended consequences",
            "Resource constraints prohibitive",
            "Conflicts with existing systems",
            "Implementation risks too high",
            "Alternative approaches preferable",
        ]
        arguments_modify = [
            "Needs phased implementation",
            "Requires additional safeguards",
            "Scope should be narrowed",
            "Timeline needs adjustment",
        ]

        # Run discussion rounds (2 exchanges per round)
        for round_num in range(max_rounds * 2):
            sender = random.choice(agents)
            recipient = random.choice([a for a in agents if a.id != sender.id])

            self.dashboard.set_agent_status(sender.id, "speaking")

            # Calculate influence based on sender confidence and recipient openness
            influence_strength = sender.belief_confidence * random.uniform(0.1, 0.3)

            # Recipient belief moves toward sender
            old_pos = recipient.belief_position
            new_pos = old_pos + (sender.belief_position - old_pos) * influence_strength

            # Confidence changes based on agreement
            agreement = 1 - abs(sender.belief_position - old_pos)
            new_conf = recipient.belief_confidence + (agreement - 0.5) * 0.1

            # Choose argument based on sender's position
            if sender.belief_position > 0.3:
                argument = random.choice(arguments_for)
                stance = "for"
            elif sender.belief_position < -0.3:
                argument = random.choice(arguments_against)
                stance = "against"
            else:
                argument = random.choice(arguments_modify)
                stance = "modify"

            # Record key argument in context
            self.dashboard.context.key_arguments.append((sender.id, argument, stance))
            # Keep only last 10 arguments
            if len(self.dashboard.context.key_arguments) > 10:
                self.dashboard.context.key_arguments.pop(0)

            # Set agent's current argument for display
            sender.current_argument = argument

            msg = f"[{stance.upper()}] {argument}"
            self.dashboard.add_message(sender.id, recipient.id, "respond", msg)
            self.dashboard.update_agent_belief(recipient.id, new_pos, new_conf, influencer=sender.id)

            self.dashboard.set_agent_status(sender.id, "thinking", f"Argued: {argument[:25]}...")
            await asyncio.sleep(0.25)

        # Phase 4: Voting with multiple rounds - can exit early on consensus
        voting_round = 0
        continue_discussion = True

        while continue_discussion and voting_round < max_voting_rounds:
            voting_round += 1
            is_final_round = (voting_round >= max_voting_rounds)

            self.dashboard.deliberation_phase = f"voting (round {voting_round}/{max_voting_rounds})"
            self.dashboard.consensus = {"Support": 0, "Oppose": 0, "Modify": 0, "Abstain": 0, "Continue": 0}

            any_continue = False

            for agent in agents:
                self.dashboard.set_agent_status(agent.id, "voting")

                # Vote based on belief position
                pos = agent.belief_position
                conf = agent.belief_confidence

                # Low confidence agents may vote to continue (unless final round)
                if not is_final_round and conf < 0.5 and random.random() < 0.3:
                    vote = "Continue"
                    any_continue = True
                elif pos > 0.4:
                    vote = "Support"
                elif pos > 0:
                    vote = "Modify"
                elif pos > -0.4:
                    vote = "Abstain"
                else:
                    vote = "Oppose"

                self.dashboard.consensus[vote] += 1
                self.dashboard.set_agent_vote(agent.id, "main_proposal", vote)
                self.dashboard.add_message(agent.id, "all", "vote", f"Vote: {vote} (belief: {pos:+.2f}, conf: {conf:.0%})")

                # Lock in confidence after voting
                self.dashboard.update_agent_belief(agent.id, pos, min(1.0, conf + 0.1))
                await asyncio.sleep(0.15)

            # Check if we should continue
            if any_continue and not is_final_round:
                # Quick discussion round before next vote
                self.dashboard.deliberation_phase = "addressing concerns"
                for _ in range(2):
                    sender = random.choice(agents)
                    recipient = random.choice([a for a in agents if a.id != sender.id])
                    self.dashboard.set_agent_status(sender.id, "speaking")
                    self.dashboard.add_message(sender.id, recipient.id, "respond", "Addressing remaining concerns...")
                    # Boost confidence slightly
                    recipient.belief_confidence = min(1.0, recipient.belief_confidence + 0.1)
                    await asyncio.sleep(0.2)
                continue_discussion = True
            else:
                # No one wants to continue or final round - exit
                continue_discussion = False

        # Phase 5: Synthesis
        self.dashboard.deliberation_phase = "synthesis"
        synthesizer = max(agents, key=lambda a: a.belief_confidence)
        self.dashboard.set_agent_status(synthesizer.id, "speaking")

        # Calculate final consensus
        avg_belief = sum(a.belief_position for a in agents) / len(agents)
        decision = "APPROVED" if avg_belief > 0.2 else "MODIFIED" if avg_belief > -0.2 else "REJECTED"

        self.dashboard.set_agent_status(synthesizer.id, "thinking", f"Consensus: {decision} (μ={avg_belief:+.2f})")

        await asyncio.sleep(0.5)

        self.dashboard.add_message(
            synthesizer.id, "all", "broadcast",
            f"Consensus reached: {decision} (avg belief: {avg_belief:+.2f})"
        )

        # Mark all as done
        self.dashboard.deliberation_phase = "complete"
        for agent in agents:
            self.dashboard.set_agent_status(agent.id, "done")

        await asyncio.sleep(2)

    async def run_continuous(self, max_rounds: int = 5, max_voting_rounds: int = 3, continuous: bool = False):
        """Run swarm deliberation(s) with rich topic context.

        Args:
            max_rounds: Maximum discussion rounds per deliberation
            max_voting_rounds: Maximum voting rounds before forcing decision
            continuous: If True, loop forever picking new topics. If False, run once and exit.
        """
        # Structured topics with descriptions and criteria
        topic_configs = [
            {
                "topic": "Should AI systems self-improve autonomously?",
                "description": "Evaluating recursive self-improvement without human oversight",
                "criterion": "supermajority",
            },
            {
                "topic": "How do we ensure AI alignment with human values?",
                "description": "Mechanisms for value learning and goal stability",
                "criterion": "consensus",
            },
            {
                "topic": "What governance structures for multi-agent systems?",
                "description": "Hierarchical vs flat decision-making architectures",
                "criterion": "majority",
            },
            {
                "topic": "Distributed consensus in heterogeneous swarms",
                "description": "Byzantine fault tolerance with diverse agent capabilities",
                "criterion": "supermajority",
            },
            {
                "topic": "Emergent behavior vs explicit coordination",
                "description": "Trade-offs between adaptability and predictability",
                "criterion": "plurality",
            },
            {
                "topic": "Trust mechanisms in agent networks",
                "description": "Reputation systems and cryptographic verification",
                "criterion": "majority",
            },
            {
                "topic": "Should agents have veto power on critical decisions?",
                "description": "Individual vs collective authority in high-stakes scenarios",
                "criterion": "consensus",
            },
            {
                "topic": "Optimal resource allocation across agent pools",
                "description": "Fairness vs efficiency in compute distribution",
                "criterion": "majority",
            },
        ]

        while True:
            config = random.choice(topic_configs)
            await self.run_deliberation(
                topic=config["topic"],
                description=config["description"],
                criterion=config["criterion"],
                max_rounds=max_rounds,
                max_voting_rounds=max_voting_rounds,
            )
            if not continuous:
                break
            await asyncio.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """Run the swarm dashboard."""
    console.print("\n[bold #7C3AED]╔══════════════════════════════════════════════════════╗[/]")
    console.print("[bold #7C3AED]║[/]   [bold white]LIDA Multi-Agent Swarm Dashboard[/]               [bold #7C3AED]║[/]")
    console.print("[bold #7C3AED]╚══════════════════════════════════════════════════════╝[/]\n")

    # Parse args
    import sys
    num_agents = 8
    use_llm = False
    max_rounds = 5
    max_voting_rounds = 3
    continuous = False

    for arg in sys.argv[1:]:
        if arg.startswith("--agents="):
            num_agents = int(arg.split("=")[1])
        elif arg.startswith("--max-rounds="):
            max_rounds = int(arg.split("=")[1])
        elif arg.startswith("--max-voting-rounds="):
            max_voting_rounds = int(arg.split("=")[1])
        elif arg == "--llm":
            use_llm = True
        elif arg == "--continuous":
            continuous = True

    console.print(f"[cyan]Initializing swarm with {num_agents} agents...[/]")
    console.print(f"[dim]Max rounds: {max_rounds}, Max voting rounds: {max_voting_rounds}, Continuous: {continuous}[/]")

    # Create dashboard and simulator
    dashboard = SwarmDashboard(num_agents=num_agents)
    simulator = SwarmSimulator(dashboard, use_llm=use_llm)

    if use_llm:
        await simulator.initialize_llm()

    console.print("[green]✓ Swarm initialized[/]")
    console.print("[dim]Starting live dashboard in 2 seconds...[/]\n")
    await asyncio.sleep(2)

    # Run simulation in background
    sim_task = asyncio.create_task(simulator.run_continuous(
        max_rounds=max_rounds,
        max_voting_rounds=max_voting_rounds,
        continuous=continuous,
    ))

    # Run dashboard
    try:
        with Live(
            dashboard.make_layout(),
            console=console,
            refresh_per_second=8,
            screen=True,
        ) as live:
            while not sim_task.done():
                live.update(dashboard.make_layout())
                await asyncio.sleep(0.125)
            # Show final state briefly
            live.update(dashboard.make_layout())
            await asyncio.sleep(2)
    except KeyboardInterrupt:
        sim_task.cancel()
    finally:
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    asyncio.run(main())
