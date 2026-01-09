#!/usr/bin/env python3
"""
HYPERDASH - Ultra-Dense Real-Time Multi-Agent Dashboard

Maximum data density visualization featuring:
- Animated waveform activity indicators
- Live token flow sparklines with gradients
- Dependency DAG with critical path glow
- Heatmap-style agent utilization matrix
- Streaming thought bubbles with typing animation
- Cost burn ticker with projection
- Latency histogram bars
- Network pulse visualization
- Multi-panel synchronized animations
"""

import asyncio
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import heapq

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box
from rich.style import Style
from rich.align import Align

console = Console(force_terminal=True, color_system="truecolor")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Animation frames
SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
WAVE = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█", "▇", "▆", "▅", "▄", "▃", "▂"]
PULSE = ["○", "◎", "●", "◉", "●", "◎"]
DOTS = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
BRAILLE_WAVE = ["⠁", "⠂", "⠄", "⡀", "⢀", "⠠", "⠐", "⠈"]
BLOCKS = " ▏▎▍▌▋▊▉█"
SPARK = "▁▂▃▄▅▆▇█"
SHADE = " ░▒▓█"
FIRE = [".", ":", "^", "*", "x", "s", "S", "#", "$", "@"]
VERTICAL_BARS = "▁▂▃▄▅▆▇█"

# Color palettes
HEAT_COLORS = ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560", "#ff6b6b", "#feca57", "#fff"]
GRADIENT_CYAN = ["#0d1b2a", "#1b263b", "#415a77", "#778da9", "#e0e1dd"]
GRADIENT_PURPLE = ["#10002b", "#240046", "#3c096c", "#5a189a", "#7b2cbf", "#9d4edd", "#c77dff", "#e0aaff"]
GRADIENT_GREEN = ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6"]
NEON = {"cyan": "#00fff5", "pink": "#ff00ff", "yellow": "#ffff00", "green": "#00ff00", "orange": "#ff6600"}

# Status configuration
class TaskState(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    BLOCKED = "blocked"
    RUNNING = "running"
    STREAMING = "streaming"
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"

STATE_VIZ = {
    TaskState.PENDING:   {"sym": "○", "color": "#4a5568", "glow": False, "anim": None},
    TaskState.QUEUED:    {"sym": "◌", "color": "#63b3ed", "glow": False, "anim": "pulse"},
    TaskState.BLOCKED:   {"sym": "◈", "color": "#fc8181", "glow": True,  "anim": "shake"},
    TaskState.RUNNING:   {"sym": "●", "color": "#68d391", "glow": True,  "anim": "spin"},
    TaskState.STREAMING: {"sym": "◉", "color": "#b794f4", "glow": True,  "anim": "wave"},
    TaskState.THINKING:  {"sym": "◐", "color": "#f6ad55", "glow": True,  "anim": "think"},
    TaskState.TOOL_USE:  {"sym": "⚙", "color": "#fbd38d", "glow": True,  "anim": "gear"},
    TaskState.SUCCESS:   {"sym": "✓", "color": "#68d391", "glow": False, "anim": None},
    TaskState.FAILED:    {"sym": "✗", "color": "#fc8181", "glow": True,  "anim": "flash"},
    TaskState.RETRY:     {"sym": "↻", "color": "#f6ad55", "glow": True,  "anim": "spin"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenFlow:
    """Token flow tracking with time series."""
    input_tokens: int = 0
    output_tokens: int = 0
    input_history: deque = field(default_factory=lambda: deque(maxlen=60))
    output_history: deque = field(default_factory=lambda: deque(maxlen=60))
    cost: float = 0.0
    cost_history: deque = field(default_factory=lambda: deque(maxlen=60))

    def record(self, inp: int = 0, out: int = 0):
        self.input_tokens += inp
        self.output_tokens += out
        self.input_history.append(inp)
        self.output_history.append(out)
        # Estimate cost (claude-3-sonnet rates)
        cost_delta = (inp * 0.003 + out * 0.015) / 1000
        self.cost += cost_delta
        self.cost_history.append(cost_delta)


@dataclass
class LatencyStats:
    """Latency tracking with histogram."""
    samples: deque = field(default_factory=lambda: deque(maxlen=100))
    buckets: List[int] = field(default_factory=lambda: [0]*10)  # 0-100ms buckets

    def record(self, ms: float):
        self.samples.append(ms)
        bucket = min(9, int(ms / 100))
        self.buckets[bucket] += 1

    def histogram_bars(self, width: int = 10) -> str:
        if not any(self.buckets):
            return "░" * width
        max_v = max(self.buckets)
        return "".join(VERTICAL_BARS[min(7, int(b / max_v * 7))] if max_v > 0 else "▁"
                      for b in self.buckets[:width])


@dataclass
class StreamChunk:
    ts: float
    content: str
    chunk_type: str = "text"
    tokens: int = 0


@dataclass
class TaskNode:
    id: str
    name: str
    task_type: str
    state: TaskState = TaskState.PENDING

    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0

    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: float = 0.0

    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)

    agent_id: Optional[str] = None
    model: str = "sonnet"

    stream_buffer: deque = field(default_factory=lambda: deque(maxlen=100))
    thinking_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    current_thought: str = ""
    thought_progress: int = 0  # For typing animation

    tokens: TokenFlow = field(default_factory=TokenFlow)
    latency: LatencyStats = field(default_factory=LatencyStats)

    tool_name: Optional[str] = None
    error: Optional[str] = None
    result: Optional[str] = None

    collapsed: bool = False
    heat: float = 0.0  # Activity heat 0-1
    priority: int = 0

    def duration(self) -> float:
        if self.started_at:
            return (self.completed_at or time.time()) - self.started_at
        return 0.0

    def is_active(self) -> bool:
        return self.state in (TaskState.RUNNING, TaskState.STREAMING,
                             TaskState.THINKING, TaskState.TOOL_USE, TaskState.RETRY)


@dataclass
class Agent:
    id: str
    name: str
    color: str
    symbol: str

    state: str = "idle"
    current_task: Optional[str] = None

    completed: int = 0
    failed: int = 0
    tokens: int = 0
    cost: float = 0.0

    utilization_history: deque = field(default_factory=lambda: deque(maxlen=30))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=30))


@dataclass
class Event:
    ts: float
    event_type: str
    source: str
    content: str
    severity: str = "info"


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERDASH CORE
# ═══════════════════════════════════════════════════════════════════════════════

class HyperDash:
    """Ultra-dense animated dashboard."""

    def __init__(self):
        self.tasks: Dict[str, TaskNode] = {}
        self.roots: List[str] = []
        self.agents: Dict[str, Agent] = {}
        self.events: deque = deque(maxlen=500)
        self.global_tokens = TokenFlow()
        self.global_latency = LatencyStats()

        self.start_time = time.time()
        self.frame = 0
        self.selected_task: Optional[str] = None

        # Animation state
        self.wave_offset = 0
        self.pulse_phase = 0
        self.glow_intensity = 0.5
        self.scroll_offset = 0

        # Counters
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.active_tasks = 0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    # ─────────────────────────────────────────────────────────────────────────
    # Visual Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _animated_symbol(self, state: TaskState) -> str:
        cfg = STATE_VIZ[state]
        anim = cfg["anim"]

        if anim == "spin":
            return SPINNERS[self.frame % len(SPINNERS)]
        elif anim == "wave":
            return DOTS[self.frame % len(DOTS)]
        elif anim == "pulse":
            return PULSE[self.frame % len(PULSE)]
        elif anim == "think":
            return ["◐", "◓", "◑", "◒"][self.frame % 4]
        elif anim == "gear":
            return ["⚙", "⚙", "⛭", "⛭"][(self.frame // 2) % 4]
        elif anim == "flash":
            return "✗" if self.frame % 4 < 2 else " "
        elif anim == "shake":
            return ["◈", "◇"][(self.frame // 2) % 2]
        return cfg["sym"]

    def _heat_color(self, heat: float) -> str:
        """Get color from heat gradient."""
        idx = min(len(HEAT_COLORS) - 1, int(heat * (len(HEAT_COLORS) - 1)))
        return HEAT_COLORS[idx]

    def _mini_sparkline(self, values: List[float], width: int = 8) -> str:
        if not values:
            return "·" * width
        vals = list(values)[-width:]
        if len(vals) < width:
            vals = [0] * (width - len(vals)) + vals
        max_v = max(vals) if vals else 1
        if max_v == 0:
            return SPARK[0] * width
        return "".join(SPARK[min(7, int(v / max_v * 7))] for v in vals)

    def _progress_bar(self, progress: float, width: int = 10) -> Text:
        filled = int(progress * width)
        partial_idx = int((progress * width - filled) * 8)

        bar = Text()
        bar.append("█" * filled, style="#68d391")
        if partial_idx > 0 and filled < width:
            bar.append(BLOCKS[partial_idx], style="#68d391")
            filled += 1
        bar.append("░" * (width - filled), style="#2d3748")
        return bar

    def _wave_bar(self, width: int = 20, amplitude: float = 1.0) -> str:
        """Generate animated wave pattern."""
        result = ""
        for i in range(width):
            phase = (self.wave_offset + i) / 3
            val = (math.sin(phase) + 1) / 2 * amplitude
            idx = min(7, int(val * 7))
            result += SPARK[idx]
        return result

    def _typing_text(self, text: str, progress: int) -> str:
        """Simulate typing animation."""
        visible = text[:progress]
        cursor = "▋" if self.frame % 6 < 3 else " "
        return visible + cursor

    def _format_tokens(self, n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 1000000:
            return f"{n/1000:.1f}K"
        return f"{n/1000000:.2f}M"

    def _format_cost(self, c: float) -> str:
        if c < 0.01:
            return f"${c:.4f}"
        if c < 1:
            return f"${c:.3f}"
        return f"${c:.2f}"

    def _format_duration(self, s: float) -> str:
        if s < 1:
            return f"{s*1000:.0f}ms"
        if s < 60:
            return f"{s:.1f}s"
        m, s = divmod(int(s), 60)
        return f"{m}m{int(s)}s"

    # ─────────────────────────────────────────────────────────────────────────
    # Task Management
    # ─────────────────────────────────────────────────────────────────────────

    def create_saga(self, name: str) -> str:
        task = TaskNode(
            id=f"S-{uuid.uuid4().hex[:6]}",
            name=name,
            task_type="saga",
        )
        self.tasks[task.id] = task
        self.roots.append(task.id)
        self.total_tasks += 1
        self._emit("saga_start", task.id, f"▶ {name}")
        return task.id

    def create_task(self, name: str, parent_id: str, depends_on: List[str] = None,
                    agent_id: str = None, model: str = "sonnet") -> str:
        parent = self.tasks.get(parent_id)
        task = TaskNode(
            id=f"T-{uuid.uuid4().hex[:6]}",
            name=name,
            task_type="task",
            parent_id=parent_id,
            depends_on=depends_on or [],
            agent_id=agent_id,
            model=model,
            depth=parent.depth + 1 if parent else 1,
        )
        self.tasks[task.id] = task
        if parent:
            parent.children.append(task.id)
        for dep in (depends_on or []):
            if dep in self.tasks:
                self.tasks[dep].blocks.append(task.id)
        self.total_tasks += 1
        return task.id

    def create_subtask(self, name: str, parent_id: str) -> str:
        parent = self.tasks.get(parent_id)
        task = TaskNode(
            id=f"s-{uuid.uuid4().hex[:6]}",
            name=name,
            task_type="subtask",
            parent_id=parent_id,
            depth=parent.depth + 1 if parent else 2,
        )
        self.tasks[task.id] = task
        if parent:
            parent.children.append(task.id)
        self.total_tasks += 1
        return task.id

    def start_task(self, task_id: str, input_tokens: int = 0):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.RUNNING
            task.started_at = time.time()
            task.tokens.record(inp=input_tokens)
            task.heat = 1.0
            self.active_tasks += 1
            self._emit("task_start", task_id, f"● {task.name}")
            if task.agent_id and task.agent_id in self.agents:
                self.agents[task.agent_id].state = "running"
                self.agents[task.agent_id].current_task = task_id

    def stream_thinking(self, task_id: str, thought: str):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.THINKING
            task.current_thought = thought
            task.thought_progress = 0
            task.thinking_buffer.append(thought)

    def stream_content(self, task_id: str, content: str, tokens: int = 0, latency_ms: float = 0):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.STREAMING
            task.stream_buffer.append(StreamChunk(time.time(), content, "text", tokens))
            task.tokens.record(out=tokens)
            task.latency.record(latency_ms)
            task.heat = min(1.0, task.heat + 0.1)
            self.global_tokens.record(out=tokens)
            self.global_latency.record(latency_ms)

    def use_tool(self, task_id: str, tool_name: str):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.TOOL_USE
            task.tool_name = tool_name
            self._emit("tool_call", task_id, f"⚙ {tool_name}")

    def complete_tool(self, task_id: str):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.STREAMING
            task.tool_name = None

    def update_progress(self, task_id: str, progress: float):
        task = self.tasks.get(task_id)
        if task:
            task.progress = max(0, min(1, progress))

    def complete_task(self, task_id: str, result: str = None):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.SUCCESS
            task.completed_at = time.time()
            task.progress = 1.0
            task.result = result
            self.completed_tasks += 1
            self.active_tasks = max(0, self.active_tasks - 1)
            self._emit("task_done", task_id, f"✓ {task.name} ({self._format_duration(task.duration())})")
            if task.agent_id and task.agent_id in self.agents:
                agent = self.agents[task.agent_id]
                agent.state = "idle"
                agent.current_task = None
                agent.completed += 1
                agent.tokens += task.tokens.input_tokens + task.tokens.output_tokens
                agent.cost += task.tokens.cost

    def fail_task(self, task_id: str, error: str):
        task = self.tasks.get(task_id)
        if task:
            task.state = TaskState.FAILED
            task.completed_at = time.time()
            task.error = error
            self.failed_tasks += 1
            self.active_tasks = max(0, self.active_tasks - 1)
            self._emit("task_fail", task_id, f"✗ {error}", "error")
            if task.agent_id and task.agent_id in self.agents:
                agent = self.agents[task.agent_id]
                agent.state = "idle"
                agent.current_task = None
                agent.failed += 1

    def add_agent(self, aid: str, name: str, color: str, symbol: str):
        self.agents[aid] = Agent(id=aid, name=name, color=color, symbol=symbol)

    def _emit(self, event_type: str, source: str, content: str, severity: str = "info"):
        self.events.append(Event(self.elapsed(), event_type, source, content, severity))

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering - Header Strip
    # ─────────────────────────────────────────────────────────────────────────

    def _render_header(self) -> Text:
        """Ultra-compact animated header."""
        t = Text()

        # Logo with pulse
        pulse = PULSE[self.frame % len(PULSE)]
        t.append(f" {pulse} ", style="bold #b794f4")
        t.append("HYPER", style="bold #00fff5")
        t.append("DASH ", style="bold white")

        # Elapsed with wave
        mins, secs = divmod(int(self.elapsed()), 60)
        t.append(f"⏱{mins:02d}:{secs:02d}", style="#63b3ed")

        # Task counters with mini bars
        active_bar = SPARK[min(7, self.active_tasks)]
        t.append(f" │ ⚡{self.active_tasks}", style="#fbd38d")
        t.append(active_bar, style="#fbd38d")

        t.append(f" ✓{self.completed_tasks}", style="#68d391")
        t.append(f" ✗{self.failed_tasks}", style="#fc8181")

        # Token flow with sparkline
        tok_spark = self._mini_sparkline(list(self.global_tokens.output_history), 6)
        total_tok = self.global_tokens.input_tokens + self.global_tokens.output_tokens
        t.append(f" │ {self._format_tokens(total_tok)}", style="#b794f4")
        t.append(f"[{tok_spark}]", style="#9f7aea")

        # Cost with burn rate
        t.append(f" {self._format_cost(self.global_tokens.cost)}", style="#68d391")
        if self.elapsed() > 0:
            rate = self.global_tokens.cost / (self.elapsed() / 60)
            t.append(f"↗{self._format_cost(rate)}/m", style="dim #68d391")

        # Latency histogram
        lat_hist = self.global_latency.histogram_bars(8)
        t.append(f" │ ⏳[{lat_hist}]", style="#f6ad55")

        # Activity wave
        wave = self._wave_bar(10, amplitude=0.5 + 0.5 * (self.active_tasks / max(1, self.total_tasks)))
        t.append(f" {wave}", style="#00fff5")

        return t

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering - Task Tree
    # ─────────────────────────────────────────────────────────────────────────

    def _render_task_node(self, task: TaskNode, is_last: bool = False) -> List[Text]:
        """Render single task with all metrics inline."""
        lines = []
        cfg = STATE_VIZ[task.state]

        # Build main line
        line = Text()

        # Tree indent
        indent = "  " * task.depth
        branch = "└─" if is_last else "├─"
        line.append(f"{indent}{branch}" if task.depth > 0 else "", style="dim #4a5568")

        # Collapse indicator
        if task.children:
            coll = "▶" if task.collapsed else "▼"
            line.append(f"{coll}", style="dim #63b3ed")
        else:
            line.append(" ")

        # Animated state symbol with glow effect
        sym = self._animated_symbol(task.state)
        sym_style = cfg["color"]
        if cfg["glow"] and self.frame % 4 < 2:
            sym_style = f"bold {cfg['color']}"
        line.append(f"{sym} ", style=sym_style)

        # Heat indicator (activity level)
        heat_char = SHADE[min(4, int(task.heat * 4))]
        heat_color = self._heat_color(task.heat)
        line.append(f"{heat_char}", style=heat_color)

        # Task type indicator
        type_sym = {"saga": "◆", "task": "○", "subtask": "·"}.get(task.task_type, "·")
        line.append(f"{type_sym}", style="dim")

        # Name (truncated)
        name_width = 18
        name = task.name[:name_width].ljust(name_width)
        name_style = ""
        if task.state == TaskState.FAILED:
            name_style = "strike dim #fc8181"
        elif task.is_active():
            name_style = "bold"
        line.append(f"{name}", style=name_style)

        # Progress bar (if active)
        if task.is_active() and task.progress > 0:
            line.append(" ")
            line.append_text(self._progress_bar(task.progress, 6))
            line.append(f"{task.progress:3.0%}", style="dim")

        # Duration
        if task.started_at:
            dur = task.duration()
            line.append(f" {self._format_duration(dur)}", style="dim")

        # Token mini-sparkline
        if task.tokens.output_history:
            spark = self._mini_sparkline(list(task.tokens.output_history), 4)
            total = task.tokens.input_tokens + task.tokens.output_tokens
            line.append(f" {self._format_tokens(total)}", style="dim #b794f4")
            line.append(spark, style="#9f7aea")

        # Cost
        if task.tokens.cost > 0:
            line.append(f" {self._format_cost(task.tokens.cost)}", style="dim #68d391")

        # Agent
        if task.agent_id and task.agent_id in self.agents:
            agent = self.agents[task.agent_id]
            line.append(f" {agent.symbol}", style=agent.color)

        # Model tag
        model_tags = {"sonnet": "S", "opus": "O", "haiku": "H"}
        line.append(f"[{model_tags.get(task.model, '?')}]", style="dim")

        # Dependency indicator
        blocked_deps = [d for d in task.depends_on
                       if d in self.tasks and self.tasks[d].state != TaskState.SUCCESS]
        if blocked_deps:
            line.append(f" ⏳{len(blocked_deps)}", style="#f6ad55")

        # Tool indicator
        if task.tool_name:
            line.append(f" ⚙{task.tool_name[:8]}", style="#fbd38d")

        # Error indicator
        if task.error:
            line.append(" ⚠", style="#fc8181")

        lines.append(line)

        # Thinking line (with typing animation)
        if task.state == TaskState.THINKING and task.current_thought:
            think_line = Text()
            think_line.append(f"{indent}   │ ", style="dim #f6ad55")
            think_line.append("💭 ", style="#f6ad55")
            # Animate typing
            task.thought_progress = min(len(task.current_thought), task.thought_progress + 2)
            visible_thought = self._typing_text(task.current_thought[:40], task.thought_progress)
            think_line.append(visible_thought, style="italic #f6ad55")
            lines.append(think_line)

        # Streaming content line
        elif task.state == TaskState.STREAMING and task.stream_buffer:
            stream_line = Text()
            stream_line.append(f"{indent}   │ ", style="dim #b794f4")
            last_chunk = task.stream_buffer[-1]
            content = last_chunk.content.replace("\n", " ")[:45]
            # Typing cursor effect
            cursor = "▋" if self.frame % 6 < 3 else ""
            stream_line.append(f"{content}{cursor}", style="#e9d8a6")
            if last_chunk.tokens > 0:
                stream_line.append(f" +{last_chunk.tokens}", style="dim")
            lines.append(stream_line)

        # Error line
        elif task.error and task.state == TaskState.FAILED:
            err_line = Text()
            err_line.append(f"{indent}   └─ ", style="dim")
            err_line.append(f"✗ {task.error[:40]}", style="italic #fc8181")
            lines.append(err_line)

        return lines

    def _render_tree(self) -> List[Text]:
        """Render full task tree."""
        lines = []

        def render_recursive(task_id: str, is_last: bool = False):
            task = self.tasks.get(task_id)
            if not task:
                return

            task_lines = self._render_task_node(task, is_last)
            lines.extend(task_lines)

            # Decay heat
            task.heat = max(0, task.heat - 0.02)

            if not task.collapsed:
                children = task.children
                for i, child_id in enumerate(children):
                    render_recursive(child_id, i == len(children) - 1)

        for i, root_id in enumerate(self.roots):
            render_recursive(root_id, i == len(self.roots) - 1)

        return lines

    def _make_tree_panel(self) -> Panel:
        lines = self._render_tree()
        if not lines:
            lines = [Text("Waiting for sagas...", style="dim italic")]

        # Limit visible lines
        visible = lines[:28]

        return Panel(
            Group(*visible),
            title=f"[bold]▼ TASKS ({self.total_tasks})[/]",
            border_style="#4a5568",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering - Activity Stream
    # ─────────────────────────────────────────────────────────────────────────

    def _make_stream_panel(self) -> Panel:
        lines = []

        type_styles = {
            "saga_start": ("◆", "#b794f4"),
            "task_start": ("▶", "#68d391"),
            "task_done": ("✓", "#68d391"),
            "task_fail": ("✗", "#fc8181"),
            "tool_call": ("⚙", "#fbd38d"),
        }

        recent = list(self.events)[-20:]
        for evt in reversed(recent):
            sym, color = type_styles.get(evt.event_type, ("·", "#4a5568"))
            line = Text()
            line.append(f"{evt.ts:6.1f}s ", style="dim")
            line.append(f"{sym} ", style=color)

            # Task name
            task = self.tasks.get(evt.source)
            source_name = task.name[:8] if task else evt.source[:8]
            line.append(f"{source_name:<8} ", style="bold" if task else "dim")

            # Content
            content_style = "#fc8181" if evt.severity == "error" else ""
            line.append(evt.content[:30], style=content_style)

            lines.append(line)

        if not lines:
            lines.append(Text("Awaiting events...", style="dim italic"))

        return Panel(
            Group(*lines),
            title=f"[bold]⚡ STREAM ({len(self.events)})[/]",
            border_style="#4a5568",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering - Agent Matrix
    # ─────────────────────────────────────────────────────────────────────────

    def _make_agents_panel(self) -> Panel:
        lines = []

        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            line = Text()

            # Status indicator
            if agent.state == "running":
                line.append(f"{SPINNERS[self.frame % len(SPINNERS)]} ", style="#68d391")
            else:
                line.append("○ ", style="dim")

            # Agent symbol and name
            line.append(f"{agent.symbol}", style=f"bold {agent.color}")
            line.append(f"{agent.name[:6]:<6}", style=agent.color)

            # Stats row
            line.append(f" {agent.completed:2d}✓", style="dim #68d391")
            line.append(f" {agent.failed:1d}✗", style="dim #fc8181")

            # Token/cost
            line.append(f" {self._format_tokens(agent.tokens):>4}", style="dim #b794f4")
            line.append(f" {self._format_cost(agent.cost)}", style="dim #68d391")

            # Utilization sparkline
            agent.utilization_history.append(1 if agent.state == "running" else 0)
            util_spark = self._mini_sparkline([float(x) for x in agent.utilization_history], 6)
            line.append(f" [{util_spark}]", style="dim")

            lines.append(line)

            # Current task
            if agent.current_task:
                task = self.tasks.get(agent.current_task)
                if task:
                    task_line = Text()
                    task_line.append("   └─ ", style="dim")
                    task_line.append(f"{task.name[:15]}", style="italic")
                    if task.progress > 0:
                        task_line.append(f" {task.progress:.0%}", style="dim")
                    lines.append(task_line)

        if not lines:
            lines.append(Text("No agents", style="dim italic"))

        return Panel(
            Group(*lines),
            title=f"[bold]🤖 AGENTS ({len(self.agents)})[/]",
            border_style="#4a5568",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering - Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def _make_metrics_panel(self) -> Panel:
        lines = []

        # Task completion rate
        if self.total_tasks > 0:
            rate = self.completed_tasks / self.total_tasks
            bar = self._progress_bar(rate, 12)
            line = Text()
            line.append("Complete ", style="dim")
            line.append_text(bar)
            line.append(f" {rate:.0%}", style="")
            lines.append(line)

        # Error rate
        if self.completed_tasks + self.failed_tasks > 0:
            err_rate = self.failed_tasks / (self.completed_tasks + self.failed_tasks)
            line = Text()
            line.append("Errors   ", style="dim")
            err_bar = "█" * int(err_rate * 12) + "░" * (12 - int(err_rate * 12))
            line.append(f"[{err_bar}]", style="#fc8181")
            line.append(f" {err_rate:.0%}", style="#fc8181")
            lines.append(line)

        lines.append(Text(""))

        # Token breakdown
        line = Text()
        line.append("Input    ", style="dim")
        line.append(f"{self._format_tokens(self.global_tokens.input_tokens)}", style="#63b3ed")
        lines.append(line)

        line = Text()
        line.append("Output   ", style="dim")
        line.append(f"{self._format_tokens(self.global_tokens.output_tokens)}", style="#b794f4")
        lines.append(line)

        lines.append(Text(""))

        # Cost projection
        if self.elapsed() > 10:
            rate_per_hour = self.global_tokens.cost / (self.elapsed() / 3600)
            line = Text()
            line.append("Burn     ", style="dim")
            line.append(f"{self._format_cost(rate_per_hour)}/hr", style="#68d391")
            lines.append(line)

        # Latency stats
        if self.global_latency.samples:
            avg_lat = sum(self.global_latency.samples) / len(self.global_latency.samples)
            line = Text()
            line.append("Latency  ", style="dim")
            line.append(f"{avg_lat:.0f}ms", style="#f6ad55")
            lines.append(line)

        return Panel(
            Group(*lines),
            title="[bold]📊 METRICS[/]",
            border_style="#4a5568",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering - Dependency View
    # ─────────────────────────────────────────────────────────────────────────

    def _make_deps_panel(self) -> Panel:
        lines = []

        # Find tasks with dependencies
        dep_tasks = [(tid, t) for tid, t in self.tasks.items()
                     if t.depends_on or t.blocks][:10]

        for tid, task in dep_tasks:
            line = Text()

            # Blockers
            if task.depends_on:
                for dep_id in task.depends_on[:2]:
                    dep = self.tasks.get(dep_id)
                    if dep:
                        dep_cfg = STATE_VIZ[dep.state]
                        line.append(f"{dep_cfg['sym']}", style=dep_cfg['color'])
                        line.append(f"{dep.name[:6]} ", style="dim")
                line.append("→ ", style="dim")

            # This task
            cfg = STATE_VIZ[task.state]
            sym = self._animated_symbol(task.state)
            line.append(f"{sym}", style=f"bold {cfg['color']}")
            line.append(f" {task.name[:10]} ", style="bold")

            # What it blocks
            if task.blocks:
                line.append("→ ", style="dim")
                for blocked_id in task.blocks[:2]:
                    blocked = self.tasks.get(blocked_id)
                    if blocked:
                        b_cfg = STATE_VIZ[blocked.state]
                        line.append(f"{b_cfg['sym']}", style=b_cfg['color'])
                        line.append(f"{blocked.name[:6]} ", style="dim")

            lines.append(line)

        if not lines:
            lines.append(Text("No dependencies", style="dim italic"))

        return Panel(
            Group(*lines),
            title="[bold]⟷ DEPS[/]",
            border_style="#4a5568",
            box=box.ROUNDED,
            padding=(0, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main Layout
    # ─────────────────────────────────────────────────────────────────────────

    def make_layout(self) -> Layout:
        self.frame += 1
        self.wave_offset += 0.5
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * math.pi)
        self.glow_intensity = 0.5 + 0.5 * math.sin(self.pulse_phase)

        layout = Layout()

        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )

        # Header
        layout["header"].update(Panel(self._render_header(), box=box.SIMPLE, padding=0))

        # Body: 60/40 split
        layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="side", ratio=2),
        )

        # Main: tree + deps
        layout["main"].split_column(
            Layout(name="tree"),
            Layout(name="deps", size=8),
        )
        layout["tree"].update(self._make_tree_panel())
        layout["deps"].update(self._make_deps_panel())

        # Side: stream + metrics + agents
        layout["side"].split_column(
            Layout(name="stream"),
            Layout(name="metrics", size=10),
            Layout(name="agents", size=8),
        )
        layout["stream"].update(self._make_stream_panel())
        layout["metrics"].update(self._make_metrics_panel())
        layout["agents"].update(self._make_agents_panel())

        # Footer
        footer = Text()
        footer.append(" HYPER", style="bold #00fff5")
        footer.append("DASH ", style="bold")
        footer.append(f"│ {datetime.now().strftime('%H:%M:%S')} ", style="dim")
        footer.append("│ ", style="dim")
        wave = self._wave_bar(15, 0.3 + 0.7 * (self.active_tasks / max(1, self.total_tasks)))
        footer.append(wave, style="#00fff5")
        footer.append(" │ Ctrl+C quit", style="dim")
        layout["footer"].update(Panel(footer, box=box.SIMPLE, padding=0))

        return layout


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class HyperSimulator:
    def __init__(self, dash: HyperDash):
        self.dash = dash
        self.thoughts = [
            "Analyzing code structure...",
            "Considering edge cases...",
            "Planning approach...",
            "Reviewing patterns...",
            "Validating assumptions...",
            "Synthesizing insights...",
            "Evaluating trade-offs...",
        ]
        self.tools = ["read_file", "search", "write", "bash", "web"]

    async def setup_agents(self):
        agents = [
            ("A1", "Scout", "#00fff5", "◆"),
            ("A2", "Coder", "#68d391", "●"),
            ("A3", "Review", "#f6ad55", "★"),
            ("A4", "Synth", "#b794f4", "◇"),
        ]
        for aid, name, color, sym in agents:
            self.dash.add_agent(aid, name, color, sym)

    async def run_task(self, task_id: str):
        task = self.dash.tasks.get(task_id)
        if not task:
            return

        self.dash.start_task(task_id, random.randint(200, 800))

        # Thinking phase
        for _ in range(random.randint(2, 4)):
            thought = random.choice(self.thoughts)
            self.dash.stream_thinking(task_id, thought)
            await asyncio.sleep(random.uniform(0.2, 0.5))

        # Maybe tool use
        if random.random() > 0.6:
            tool = random.choice(self.tools)
            self.dash.use_tool(task_id, tool)
            await asyncio.sleep(random.uniform(0.3, 0.7))
            self.dash.complete_tool(task_id)

        # Streaming phase
        chunks = random.randint(10, 25)
        for i in range(chunks):
            content = "".join(random.choices("abcdefghijklmnop ", k=random.randint(15, 40)))
            tokens = random.randint(5, 20)
            latency = random.uniform(30, 150)
            self.dash.stream_content(task_id, content, tokens, latency)
            self.dash.update_progress(task_id, (i + 1) / chunks)
            await asyncio.sleep(random.uniform(0.05, 0.15))

        # Complete or fail
        if random.random() > 0.12:
            self.dash.complete_task(task_id, "Done")
        else:
            self.dash.fail_task(task_id, random.choice(["Timeout", "Rate limit", "Context overflow"]))

    async def run_saga(self, name: str):
        saga_id = self.dash.create_saga(name)

        num_tasks = random.randint(3, 6)
        task_ids = []
        prev = None

        task_names = ["Analyze", "Search", "Plan", "Code", "Test", "Review", "Deploy"]

        for i in range(num_tasks):
            deps = [prev] if prev and random.random() > 0.4 else []
            agent = random.choice(list(self.dash.agents.keys()))
            model = random.choice(["sonnet", "haiku", "opus"])

            tid = self.dash.create_task(
                random.choice(task_names),
                saga_id,
                depends_on=deps,
                agent_id=agent,
                model=model,
            )

            # Subtasks
            for _ in range(random.randint(0, 2)):
                self.dash.create_subtask(random.choice(["Parse", "Validate", "Transform"]), tid)

            task_ids.append(tid)
            prev = tid

        # Execute
        for tid in task_ids:
            task = self.dash.tasks.get(tid)
            if not task:
                continue

            # Wait deps
            while True:
                done = all(
                    self.dash.tasks.get(d, TaskNode("", "", "")).state == TaskState.SUCCESS
                    for d in task.depends_on
                )
                failed = any(
                    self.dash.tasks.get(d, TaskNode("", "", "")).state == TaskState.FAILED
                    for d in task.depends_on
                )
                if failed:
                    self.dash.fail_task(tid, "Dep failed")
                    break
                if done:
                    break
                await asyncio.sleep(0.05)

            if task.state != TaskState.FAILED:
                await self.run_task(tid)

            await asyncio.sleep(random.uniform(0.1, 0.3))

    async def run_continuous(self):
        await self.setup_agents()

        sagas = [
            "Auth flow", "API endpoint", "Bug fix", "Refactor",
            "Tests", "Docs", "Deploy", "Review PR",
        ]

        while True:
            await self.run_saga(random.choice(sagas))
            await asyncio.sleep(random.uniform(0.5, 2.0))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    console.print("\n[bold #00fff5]╔══════════════════════════════════════════════╗[/]")
    console.print("[bold #00fff5]║[/] [bold white]HYPERDASH[/] [dim]Ultra-Dense Real-Time Dashboard[/]  [bold #00fff5]║[/]")
    console.print("[bold #00fff5]╚══════════════════════════════════════════════╝[/]\n")

    dash = HyperDash()
    sim = HyperSimulator(dash)

    console.print("[#00fff5]Initializing...[/]")
    await asyncio.sleep(0.5)
    console.print("[#68d391]✓ Ready[/]")
    await asyncio.sleep(0.5)

    sim_task = asyncio.create_task(sim.run_continuous())

    try:
        with Live(dash.make_layout(), console=console, refresh_per_second=15, screen=True) as live:
            while True:
                live.update(dash.make_layout())
                await asyncio.sleep(1/15)
    except KeyboardInterrupt:
        sim_task.cancel()
        console.print("\n[#f6ad55]Stopped.[/]")


if __name__ == "__main__":
    asyncio.run(main())
