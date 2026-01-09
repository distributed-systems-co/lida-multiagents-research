#!/usr/bin/env python3
"""
Hierarchical Real-Time Multi-Agent Dashboard v2

Ultra-detailed visualization with:
- Full streaming content display with token counts
- Latency tracking and cost estimation
- Gantt-style timeline view
- Rich dependency DAG visualization
- Real-time metrics and sparklines
- Nested task cards with full context
- Tool call tracking and results
- Collapsible multi-level tree with inline progress
"""

import asyncio
import os
import random
import time
import uuid
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import heapq

from rich.console import Console, Group, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.columns import Columns
from rich import box
from rich.style import Style
from rich.align import Align
from rich.rule import Rule
from rich.padding import Padding
from rich.markdown import Markdown

console = Console(force_terminal=True, color_system="truecolor")


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

class TaskState(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    BLOCKED = "blocked"
    RUNNING = "running"
    STREAMING = "streaming"
    WAITING_TOOL = "waiting_tool"
    WAITING_HUMAN = "waiting_human"
    RETRYING = "retrying"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ViewMode(Enum):
    TREE = "tree"
    TIMELINE = "timeline"
    DETAIL = "detail"
    DEPS = "deps"
    METRICS = "metrics"


class ChunkType(Enum):
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CODE = "code"
    MARKDOWN = "markdown"


STATE_CONFIG = {
    TaskState.PENDING:      {"symbol": "○", "color": "#6B7280", "style": "dim",  "label": "Pending"},
    TaskState.QUEUED:       {"symbol": "◌", "color": "#06B6D4", "style": "",     "label": "Queued"},
    TaskState.BLOCKED:      {"symbol": "◈", "color": "#EF4444", "style": "dim",  "label": "Blocked"},
    TaskState.RUNNING:      {"symbol": "●", "color": "#10B981", "style": "bold", "label": "Running"},
    TaskState.STREAMING:    {"symbol": "◉", "color": "#8B5CF6", "style": "bold", "label": "Streaming"},
    TaskState.WAITING_TOOL: {"symbol": "◐", "color": "#F59E0B", "style": "",     "label": "Tool Wait"},
    TaskState.WAITING_HUMAN:{"symbol": "◑", "color": "#EC4899", "style": "",     "label": "Human Wait"},
    TaskState.RETRYING:     {"symbol": "↻", "color": "#F59E0B", "style": "bold", "label": "Retrying"},
    TaskState.SUCCESS:      {"symbol": "✓", "color": "#10B981", "style": "",     "label": "Success"},
    TaskState.FAILED:       {"symbol": "✗", "color": "#EF4444", "style": "bold", "label": "Failed"},
    TaskState.CANCELLED:    {"symbol": "⊘", "color": "#6B7280", "style": "dim",  "label": "Cancelled"},
    TaskState.TIMEOUT:      {"symbol": "⏱", "color": "#EF4444", "style": "",     "label": "Timeout"},
}

SPINNERS = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
PROGRESS_BLOCKS = " ▏▎▍▌▋▊▉█"
SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"

# Cost estimation (per 1K tokens)
MODEL_COSTS = {
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "default": {"input": 0.001, "output": 0.002},
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StreamChunk:
    """A chunk of streaming content with full metadata."""
    timestamp: float
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    token_count: int = 0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCall:
    """A tool/function call with tracking."""
    id: str
    name: str
    arguments: Dict[str, Any]
    started_at: float
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class TokenMetrics:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    estimated_cost: float = 0.0

    def update(self, input_t: int = 0, output_t: int = 0, model: str = "default"):
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.total_tokens = self.input_tokens + self.output_tokens
        costs = MODEL_COSTS.get(model, MODEL_COSTS["default"])
        self.estimated_cost = (
            (self.input_tokens / 1000) * costs["input"] +
            (self.output_tokens / 1000) * costs["output"]
        )


@dataclass
class LatencyMetrics:
    """Latency tracking."""
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    chunk_latencies: List[float] = field(default_factory=list)
    avg_chunk_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0

    def record_chunk(self, latency_ms: float):
        self.chunk_latencies.append(latency_ms)
        if self.first_token_ms is None:
            self.first_token_ms = latency_ms
        if self.chunk_latencies:
            self.avg_chunk_latency = sum(self.chunk_latencies) / len(self.chunk_latencies)
            sorted_lat = sorted(self.chunk_latencies)
            n = len(sorted_lat)
            self.p50_latency = sorted_lat[int(n * 0.5)] if n > 0 else 0
            self.p95_latency = sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[-1] if n > 0 else 0
            self.p99_latency = sorted_lat[int(n * 0.99)] if n > 2 else sorted_lat[-1] if n > 0 else 0


@dataclass
class TaskNode:
    """A node in the task hierarchy with full tracking."""
    id: str
    name: str
    task_type: str  # saga, task, subtask, action
    state: TaskState = TaskState.PENDING

    # Hierarchy
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)
    depth: int = 0

    # Timing
    created_at: float = field(default_factory=time.time)
    queued_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None

    # Progress
    progress: float = 0.0
    progress_message: str = ""
    estimated_duration: Optional[float] = None

    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    blocked_by: Optional[str] = None

    # Agent & Model
    agent_id: Optional[str] = None
    model: str = "default"

    # Description & Result
    description: str = ""
    input_summary: str = ""
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    # Streaming
    stream_buffer: deque = field(default_factory=lambda: deque(maxlen=200))
    is_streaming: bool = False
    stream_complete: bool = False

    # Tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)
    active_tool: Optional[str] = None

    # Metrics
    tokens: TokenMetrics = field(default_factory=TokenMetrics)
    latency: LatencyMetrics = field(default_factory=LatencyMetrics)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    priority: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    # UI state
    collapsed: bool = False
    selected: bool = False
    highlight_until: float = 0.0

    def duration(self) -> Optional[float]:
        if self.started_at:
            end = self.completed_at or time.time()
            return end - self.started_at
        return None

    def queue_wait_time(self) -> Optional[float]:
        if self.queued_at and self.started_at:
            return self.started_at - self.queued_at
        elif self.queued_at:
            return time.time() - self.queued_at
        return None

    def is_active(self) -> bool:
        return self.state in (TaskState.RUNNING, TaskState.STREAMING,
                             TaskState.WAITING_TOOL, TaskState.RETRYING)

    def is_terminal(self) -> bool:
        return self.state in (TaskState.SUCCESS, TaskState.FAILED,
                             TaskState.CANCELLED, TaskState.TIMEOUT)


@dataclass
class QueuedTask:
    """A task in the priority queue."""
    priority: int
    enqueued_at: float
    task_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: float = 0.0
    deadline: Optional[float] = None

    def __lt__(self, other):
        # Deadline-aware priority
        if self.deadline and other.deadline:
            return self.deadline < other.deadline
        if self.deadline:
            return True
        if other.deadline:
            return False
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.enqueued_at < other.enqueued_at


@dataclass
class Agent:
    """An agent with full tracking."""
    id: str
    name: str
    agent_type: str
    color: str
    symbol: str

    # State
    state: str = "idle"
    current_task: Optional[str] = None
    current_model: str = "default"

    # Stats
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_task_duration: float = 0.0

    # Capabilities
    capabilities: Set[str] = field(default_factory=set)
    models_available: List[str] = field(default_factory=list)

    # Queue
    task_queue: List[str] = field(default_factory=list)
    max_concurrent: int = 1

    # History
    task_history: deque = field(default_factory=lambda: deque(maxlen=50))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class ActivityEvent:
    """An event in the activity stream."""
    id: str
    timestamp: float
    event_type: str
    source_id: str
    target_id: Optional[str] = None
    content: str = ""
    detail: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    causal_parent: Optional[str] = None
    severity: str = "info"  # debug, info, warning, error, critical


# ═══════════════════════════════════════════════════════════════════════════════
# TASK REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class TaskRegistry:
    """Central registry for all tasks with dependency tracking."""

    def __init__(self):
        self.tasks: Dict[str, TaskNode] = {}
        self.roots: List[str] = []
        self.by_state: Dict[TaskState, Set[str]] = defaultdict(set)
        self.by_agent: Dict[str, Set[str]] = defaultdict(set)
        self.by_tag: Dict[str, Set[str]] = defaultdict(set)

        # Dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        # Metrics
        self.total_created = 0
        self.total_completed = 0
        self.total_failed = 0

    def add_task(self, task: TaskNode) -> str:
        self.tasks[task.id] = task
        self.total_created += 1

        # Track by state
        self.by_state[task.state].add(task.id)

        # Track hierarchy
        if task.parent_id is None:
            self.roots.append(task.id)
        elif task.parent_id in self.tasks:
            parent = self.tasks[task.parent_id]
            parent.children.append(task.id)
            task.depth = parent.depth + 1

        # Track by agent
        if task.agent_id:
            self.by_agent[task.agent_id].add(task.id)

        # Track by tags
        for tag in task.tags:
            self.by_tag[tag].add(task.id)

        # Register dependencies
        for dep_id in task.depends_on:
            self.dependency_graph[task.id].add(dep_id)
            self.reverse_deps[dep_id].add(task.id)
            if dep_id in self.tasks:
                self.tasks[dep_id].blocks.append(task.id)

        return task.id

    def update_state(self, task_id: str, new_state: TaskState):
        task = self.tasks.get(task_id)
        if not task:
            return

        old_state = task.state
        self.by_state[old_state].discard(task_id)
        self.by_state[new_state].add(task_id)
        task.state = new_state

        if new_state == TaskState.SUCCESS:
            self.total_completed += 1
        elif new_state == TaskState.FAILED:
            self.total_failed += 1

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        return self.tasks.get(task_id)

    def get_children(self, task_id: str) -> List[TaskNode]:
        task = self.tasks.get(task_id)
        if not task:
            return []
        return [self.tasks[cid] for cid in task.children if cid in self.tasks]

    def get_ancestors(self, task_id: str) -> List[TaskNode]:
        ancestors = []
        task = self.tasks.get(task_id)
        while task and task.parent_id:
            parent = self.tasks.get(task.parent_id)
            if parent:
                ancestors.append(parent)
                task = parent
            else:
                break
        return ancestors

    def get_ready_tasks(self) -> List[TaskNode]:
        ready = []
        for task_id in self.by_state[TaskState.PENDING] | self.by_state[TaskState.QUEUED]:
            task = self.tasks[task_id]
            deps_satisfied = all(
                self.tasks.get(dep_id, TaskNode(id="", name="", task_type="")).state == TaskState.SUCCESS
                for dep_id in task.depends_on
            )
            if deps_satisfied:
                ready.append(task)
        return ready

    def get_blocked_chain(self, task_id: str) -> List[str]:
        """Get the chain of tasks blocking this one."""
        chain = []
        task = self.tasks.get(task_id)
        if not task:
            return chain

        visited = set()
        to_visit = list(task.depends_on)

        while to_visit:
            dep_id = to_visit.pop(0)
            if dep_id in visited:
                continue
            visited.add(dep_id)

            dep = self.tasks.get(dep_id)
            if dep and not dep.is_terminal():
                chain.append(dep_id)
                to_visit.extend(dep.depends_on)

        return chain

    def get_critical_path(self) -> List[str]:
        """Find the critical path through the task graph."""
        # Simplified: find longest chain of active/pending tasks
        def chain_length(task_id: str, visited: Set[str]) -> int:
            if task_id in visited:
                return 0
            visited.add(task_id)

            task = self.tasks.get(task_id)
            if not task or task.is_terminal():
                return 0

            max_child = 0
            for child_id in task.children:
                max_child = max(max_child, chain_length(child_id, visited))

            return 1 + max_child

        longest = []
        for root_id in self.roots:
            path = []
            current = root_id
            while current:
                task = self.tasks.get(current)
                if not task:
                    break
                path.append(current)
                if not task.children:
                    break
                # Follow the longest branch
                current = max(task.children, key=lambda c: chain_length(c, set()), default=None)

            if len(path) > len(longest):
                longest = path

        return longest


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXTUAL QUEUE
# ═══════════════════════════════════════════════════════════════════════════════

class ContextualQueue:
    """Priority queue with contextual awareness and deadline support."""

    def __init__(self):
        self.heap: List[QueuedTask] = []
        self.task_contexts: Dict[str, Dict[str, Any]] = {}
        self.waiting_on: Dict[str, Set[str]] = defaultdict(set)
        self.wait_times: Dict[str, float] = {}  # task_id -> cumulative wait time
        self.history: deque = deque(maxlen=200)

        # Stats
        self.total_enqueued = 0
        self.total_dequeued = 0
        self.avg_wait_time = 0.0
        self.max_wait_time = 0.0

    def enqueue(self, task_id: str, priority: int = 0, context: Dict = None,
                estimated_duration: float = 0, deadline: float = None):
        qt = QueuedTask(
            priority=priority,
            enqueued_at=time.time(),
            task_id=task_id,
            context=context or {},
            estimated_duration=estimated_duration,
            deadline=deadline,
        )
        heapq.heappush(self.heap, qt)
        self.task_contexts[task_id] = context or {}
        self.wait_times[task_id] = 0.0
        self.total_enqueued += 1

    def dequeue(self) -> Optional[QueuedTask]:
        while self.heap:
            qt = heapq.heappop(self.heap)
            if qt.task_id in self.task_contexts:
                wait_time = time.time() - qt.enqueued_at
                self.wait_times[qt.task_id] = wait_time
                self.max_wait_time = max(self.max_wait_time, wait_time)

                # Update average
                self.total_dequeued += 1
                self.avg_wait_time = (
                    (self.avg_wait_time * (self.total_dequeued - 1) + wait_time)
                    / self.total_dequeued
                )

                del self.task_contexts[qt.task_id]
                self.history.append(qt)
                return qt
        return None

    def peek(self, n: int = 10) -> List[QueuedTask]:
        return sorted(self.heap)[:n]

    def size(self) -> int:
        return len(self.heap)

    def get_position(self, task_id: str) -> Optional[int]:
        """Get position in queue (1-indexed)."""
        sorted_heap = sorted(self.heap)
        for i, qt in enumerate(sorted_heap):
            if qt.task_id == task_id:
                return i + 1
        return None

    def add_dependency(self, task_id: str, waits_for: str):
        self.waiting_on[task_id].add(waits_for)

    def resolve_dependency(self, completed_id: str) -> List[str]:
        unblocked = []
        for task_id, deps in list(self.waiting_on.items()):
            if completed_id in deps:
                deps.remove(completed_id)
                if not deps:
                    unblocked.append(task_id)
                    del self.waiting_on[task_id]
        return unblocked


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVITY STREAM
# ═══════════════════════════════════════════════════════════════════════════════

class ActivityStream:
    """Real-time activity stream with causal linking and filtering."""

    def __init__(self, max_events: int = 1000):
        self.events: deque = deque(maxlen=max_events)
        self.by_source: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.by_type: Dict[str, deque] = defaultdict(lambda: deque(maxlen=200))
        self.causal_chains: Dict[str, List[str]] = defaultdict(list)
        self.start_time = time.time()
        self._counter = 0

        # Event type icons
        self.type_config = {
            "saga_created":     {"icon": "◆", "color": "#7C3AED", "severity": "info"},
            "saga_completed":   {"icon": "◇", "color": "#7C3AED", "severity": "info"},
            "task_created":     {"icon": "+", "color": "#06B6D4", "severity": "debug"},
            "task_started":     {"icon": "▶", "color": "#10B981", "severity": "info"},
            "task_completed":   {"icon": "✓", "color": "#10B981", "severity": "info"},
            "task_failed":      {"icon": "✗", "color": "#EF4444", "severity": "error"},
            "task_blocked":     {"icon": "◈", "color": "#EF4444", "severity": "warning"},
            "task_unblocked":   {"icon": "⟳", "color": "#06B6D4", "severity": "info"},
            "task_retry":       {"icon": "↻", "color": "#F59E0B", "severity": "warning"},
            "stream_start":     {"icon": "│", "color": "#8B5CF6", "severity": "debug"},
            "stream_chunk":     {"icon": "·", "color": "#8B5CF6", "severity": "debug"},
            "stream_end":       {"icon": "┘", "color": "#8B5CF6", "severity": "debug"},
            "tool_call":        {"icon": "⚙", "color": "#F59E0B", "severity": "info"},
            "tool_result":      {"icon": "⚡", "color": "#10B981", "severity": "info"},
            "tool_error":       {"icon": "⚠", "color": "#EF4444", "severity": "error"},
            "dependency":       {"icon": "→", "color": "#6B7280", "severity": "debug"},
            "error":            {"icon": "✗", "color": "#EF4444", "severity": "error"},
            "warning":          {"icon": "⚠", "color": "#F59E0B", "severity": "warning"},
            "metric":           {"icon": "📊", "color": "#06B6D4", "severity": "debug"},
        }

    def emit(self, event_type: str, source_id: str, content: str = "",
             detail: str = "", target_id: str = None, metadata: Dict = None,
             causal_parent: str = None, severity: str = None) -> str:
        self._counter += 1
        event_id = f"evt-{self._counter:08d}"

        config = self.type_config.get(event_type, {"icon": "·", "color": "#6B7280", "severity": "info"})

        event = ActivityEvent(
            id=event_id,
            timestamp=time.time() - self.start_time,
            event_type=event_type,
            source_id=source_id,
            target_id=target_id,
            content=content,
            detail=detail,
            metadata=metadata or {},
            causal_parent=causal_parent,
            severity=severity or config["severity"],
        )

        self.events.append(event)
        self.by_source[source_id].append(event)
        self.by_type[event_type].append(event)

        if causal_parent:
            self.causal_chains[causal_parent].append(event_id)

        return event_id

    def get_recent(self, n: int = 30, filter_type: str = None,
                   min_severity: str = "debug") -> List[ActivityEvent]:
        severity_order = {"debug": 0, "info": 1, "warning": 2, "error": 3, "critical": 4}
        min_sev = severity_order.get(min_severity, 0)

        events = list(self.events)
        if filter_type:
            events = [e for e in events if e.event_type == filter_type]

        events = [e for e in events if severity_order.get(e.severity, 0) >= min_sev]
        return events[-n:]

    def get_for_task(self, task_id: str, n: int = 50) -> List[ActivityEvent]:
        return list(self.by_source[task_id])[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemMetrics:
    """System-wide metrics with time series."""
    # Counters
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_tasks: int = 0

    # Tokens
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0

    # Latency
    avg_task_duration: float = 0.0
    avg_first_token_latency: float = 0.0
    avg_queue_wait: float = 0.0

    # Time series (last 60 samples)
    tasks_per_minute: deque = field(default_factory=lambda: deque(maxlen=60))
    tokens_per_second: deque = field(default_factory=lambda: deque(maxlen=60))
    active_tasks_history: deque = field(default_factory=lambda: deque(maxlen=60))
    error_rate_history: deque = field(default_factory=lambda: deque(maxlen=60))
    latency_history: deque = field(default_factory=lambda: deque(maxlen=60))

    def record_sample(self, active: int, tokens_sec: float, error_rate: float, latency: float):
        self.active_tasks_history.append(active)
        self.tokens_per_second.append(tokens_sec)
        self.error_rate_history.append(error_rate)
        self.latency_history.append(latency)


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

class HierarchicalDashboard:
    """Ultra-detailed hierarchical real-time dashboard."""

    def __init__(self):
        self.registry = TaskRegistry()
        self.queue = ContextualQueue()
        self.stream = ActivityStream()
        self.agents: Dict[str, Agent] = {}
        self.metrics = SystemMetrics()

        self.start_time = time.time()
        self.frame = 0
        self.view_mode = ViewMode.TREE
        self.selected_task: Optional[str] = None
        self.detail_task: Optional[str] = None

        # View state
        self.tree_scroll = 0
        self.stream_scroll = 0
        self.show_completed = True
        self.show_subtasks = True
        self.filter_tags: Set[str] = set()
        self.min_severity = "info"

        # Animation
        self.pulse_tasks: Set[str] = set()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        e = self.elapsed()
        mins, secs = divmod(int(e), 60)
        return f"{mins:02d}:{secs:02d}"

    # ─────────────────────────────────────────────────────────────────────────
    # Task Management
    # ─────────────────────────────────────────────────────────────────────────

    def create_saga(self, name: str, description: str = "", tags: Set[str] = None,
                    metadata: Dict = None) -> str:
        task = TaskNode(
            id=f"saga-{uuid.uuid4().hex[:8]}",
            name=name,
            task_type="saga",
            description=description,
            tags=tags or set(),
            metadata=metadata or {},
        )
        self.registry.add_task(task)
        self.stream.emit("saga_created", task.id, f"Saga: {name}", description)
        self.metrics.total_tasks += 1
        self.pulse_tasks.add(task.id)
        return task.id

    def create_task(self, name: str, parent_id: str, depends_on: List[str] = None,
                    agent_id: str = None, priority: int = 0, tags: Set[str] = None,
                    description: str = "", input_summary: str = "",
                    estimated_duration: float = None, model: str = "default") -> str:
        task = TaskNode(
            id=f"task-{uuid.uuid4().hex[:8]}",
            name=name,
            task_type="task",
            parent_id=parent_id,
            depends_on=depends_on or [],
            agent_id=agent_id,
            priority=priority,
            tags=tags or set(),
            description=description,
            input_summary=input_summary,
            estimated_duration=estimated_duration,
            model=model,
        )
        self.registry.add_task(task)

        if depends_on:
            for dep in depends_on:
                self.queue.add_dependency(task.id, dep)
                self.stream.emit("dependency", task.id, f"→ {dep}", target_id=dep)

        self.metrics.total_tasks += 1
        self.pulse_tasks.add(task.id)
        return task.id

    def create_subtask(self, name: str, parent_id: str, description: str = "") -> str:
        task = TaskNode(
            id=f"sub-{uuid.uuid4().hex[:8]}",
            name=name,
            task_type="subtask",
            parent_id=parent_id,
            description=description,
        )
        self.registry.add_task(task)
        self.metrics.total_tasks += 1
        return task.id

    def start_task(self, task_id: str, input_tokens: int = 0):
        task = self.registry.get_task(task_id)
        if not task:
            return

        task.started_at = time.time()
        task.tokens.input_tokens = input_tokens
        self.registry.update_state(task_id, TaskState.RUNNING)
        self.stream.emit("task_started", task_id, f"Started: {task.name}")
        self.metrics.active_tasks += 1
        self.pulse_tasks.add(task_id)

        if task.agent_id and task.agent_id in self.agents:
            agent = self.agents[task.agent_id]
            agent.state = "running"
            agent.current_task = task_id

    def stream_to_task(self, task_id: str, content: str, chunk_type: ChunkType = ChunkType.TEXT,
                       token_count: int = 0, latency_ms: float = 0):
        task = self.registry.get_task(task_id)
        if not task:
            return

        if task.state != TaskState.STREAMING:
            self.registry.update_state(task_id, TaskState.STREAMING)
            task.is_streaming = True
            self.stream.emit("stream_start", task_id, "Streaming started")

        chunk = StreamChunk(
            timestamp=time.time(),
            content=content,
            chunk_type=chunk_type,
            token_count=token_count,
            latency_ms=latency_ms,
        )
        task.stream_buffer.append(chunk)
        task.tokens.output_tokens += token_count
        task.latency.record_chunk(latency_ms)

        # Update progress message
        if chunk_type == ChunkType.THINKING:
            task.progress_message = f"Thinking: {content[:30]}..."
        elif chunk_type == ChunkType.TEXT:
            task.progress_message = content[:40] + "..." if len(content) > 40 else content

    def add_tool_call(self, task_id: str, tool_name: str, arguments: Dict[str, Any]) -> str:
        task = self.registry.get_task(task_id)
        if not task:
            return ""

        tool_call = ToolCall(
            id=f"tool-{uuid.uuid4().hex[:8]}",
            name=tool_name,
            arguments=arguments,
            started_at=time.time(),
        )
        task.tool_calls.append(tool_call)
        task.active_tool = tool_name
        self.registry.update_state(task_id, TaskState.WAITING_TOOL)

        args_summary = ", ".join(f"{k}={str(v)[:20]}" for k, v in list(arguments.items())[:3])
        self.stream.emit("tool_call", task_id, f"⚙ {tool_name}({args_summary})")

        return tool_call.id

    def complete_tool_call(self, task_id: str, tool_id: str, result: Any = None, error: str = None):
        task = self.registry.get_task(task_id)
        if not task:
            return

        for tc in task.tool_calls:
            if tc.id == tool_id:
                tc.completed_at = time.time()
                tc.duration_ms = (tc.completed_at - tc.started_at) * 1000
                tc.result = result
                tc.error = error
                break

        task.active_tool = None

        if error:
            self.stream.emit("tool_error", task_id, f"⚠ {error}", severity="error")
        else:
            result_preview = str(result)[:50] if result else "OK"
            self.stream.emit("tool_result", task_id, f"⚡ {result_preview}")

        self.registry.update_state(task_id, TaskState.STREAMING)

    def update_progress(self, task_id: str, progress: float, message: str = ""):
        task = self.registry.get_task(task_id)
        if task:
            task.progress = max(0.0, min(1.0, progress))
            if message:
                task.progress_message = message

    def complete_task(self, task_id: str, result: str = None, output_tokens: int = 0):
        task = self.registry.get_task(task_id)
        if not task:
            return

        task.completed_at = time.time()
        task.result = result
        task.progress = 1.0
        task.is_streaming = False
        task.stream_complete = True
        task.tokens.output_tokens = output_tokens or task.tokens.output_tokens
        task.tokens.update(task.tokens.input_tokens, task.tokens.output_tokens, task.model)
        task.latency.total_ms = (task.completed_at - task.started_at) * 1000

        self.registry.update_state(task_id, TaskState.SUCCESS)
        self.metrics.completed_tasks += 1
        self.metrics.active_tasks = max(0, self.metrics.active_tasks - 1)
        self.metrics.total_cost += task.tokens.estimated_cost

        duration = task.duration() or 0
        self.stream.emit("task_completed", task_id,
                        f"✓ {task.name} ({duration:.1f}s, {task.tokens.total_tokens} tok, ${task.tokens.estimated_cost:.4f})")

        # Unblock dependent tasks
        unblocked = self.queue.resolve_dependency(task_id)
        for uid in unblocked:
            self.stream.emit("task_unblocked", uid, f"Unblocked by {task_id}")
            self.pulse_tasks.add(uid)

        if task.agent_id and task.agent_id in self.agents:
            agent = self.agents[task.agent_id]
            agent.state = "idle"
            agent.current_task = None
            agent.tasks_completed += 1
            agent.total_tokens += task.tokens.total_tokens
            agent.total_cost += task.tokens.estimated_cost

    def fail_task(self, task_id: str, error: str, retry: bool = False):
        task = self.registry.get_task(task_id)
        if not task:
            return

        if retry and task.retry_count < task.max_retries:
            task.retry_count += 1
            task.error = error
            self.registry.update_state(task_id, TaskState.RETRYING)
            self.stream.emit("task_retry", task_id,
                           f"↻ Retry {task.retry_count}/{task.max_retries}: {error}",
                           severity="warning")
            return

        task.completed_at = time.time()
        task.error = error
        task.is_streaming = False

        self.registry.update_state(task_id, TaskState.FAILED)
        self.metrics.failed_tasks += 1
        self.metrics.active_tasks = max(0, self.metrics.active_tasks - 1)

        self.stream.emit("task_failed", task_id, f"✗ {error}", severity="error")

        if task.agent_id and task.agent_id in self.agents:
            agent = self.agents[task.agent_id]
            agent.state = "idle"
            agent.current_task = None
            agent.tasks_failed += 1

    # ─────────────────────────────────────────────────────────────────────────
    # Agent Management
    # ─────────────────────────────────────────────────────────────────────────

    def add_agent(self, agent_id: str, name: str, agent_type: str,
                  color: str, symbol: str, capabilities: Set[str] = None,
                  models: List[str] = None):
        self.agents[agent_id] = Agent(
            id=agent_id,
            name=name,
            agent_type=agent_type,
            color=color,
            symbol=symbol,
            capabilities=capabilities or set(),
            models_available=models or ["default"],
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Rendering Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _sparkline(self, values: List[float], width: int = 10) -> str:
        if not values:
            return "·" * width
        values = list(values)[-width:]
        if len(values) < width:
            values = [0] * (width - len(values)) + values
        if max(values) == min(values):
            return SPARKLINE_CHARS[4] * width
        min_v, max_v = min(values), max(values)
        line = ""
        for v in values:
            idx = int((v - min_v) / (max_v - min_v) * 7) if max_v != min_v else 4
            line += SPARKLINE_CHARS[idx]
        return line

    def _progress_bar(self, progress: float, width: int = 12, color: str = "#10B981") -> Text:
        filled = int(progress * width)
        partial = int((progress * width - filled) * 8)
        bar = Text()
        bar.append("█" * filled, style=color)
        if partial > 0 and filled < width:
            bar.append(PROGRESS_BLOCKS[partial], style=color)
            filled += 1
        bar.append("░" * (width - filled), style="dim")
        return bar

    def _format_duration(self, seconds: float) -> str:
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m{secs}s"

    def _format_tokens(self, count: int) -> str:
        if count < 1000:
            return str(count)
        elif count < 1000000:
            return f"{count/1000:.1f}K"
        else:
            return f"{count/1000000:.2f}M"

    # ─────────────────────────────────────────────────────────────────────────
    # Tree View Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _render_task_node(self, task: TaskNode, depth: int = 0) -> List[Text]:
        """Render a task node with full detail."""
        lines = []
        cfg = STATE_CONFIG[task.state]
        symbol = cfg["symbol"]
        color = cfg["color"]
        style = cfg["style"]

        # Animated spinner for active states
        if task.is_active():
            symbol = SPINNERS[self.frame % len(SPINNERS)]

        # Pulse effect for recently updated tasks
        is_pulsing = task.id in self.pulse_tasks and self.frame % 4 < 2

        # Main task line
        line = Text()

        # Indent with tree characters
        if depth > 0:
            indent = "  " * (depth - 1)
            connector = "├─" if task.children else "├─"
            line.append(f"{indent}{connector} ", style="dim")

        # Collapse indicator
        if task.children:
            collapse = "▶" if task.collapsed else "▼"
            line.append(f"{collapse} ", style="dim cyan")
        else:
            line.append("  ")

        # State symbol
        sym_style = f"{style} {color}" if style else color
        if is_pulsing:
            sym_style = f"bold reverse {color}"
        line.append(f"{symbol} ", style=sym_style)

        # Task type indicator
        type_icons = {"saga": "◆", "task": "○", "subtask": "·"}
        line.append(f"{type_icons.get(task.task_type, '·')} ", style="dim")

        # Task name
        name_width = 25
        name = task.name[:name_width].ljust(name_width)
        name_style = style if style else ""
        if task.state == TaskState.FAILED:
            name_style = "strike dim red"
        elif task.selected:
            name_style = "bold reverse"
        line.append(name, style=name_style)

        # Progress bar for active tasks
        if task.is_active() and task.progress > 0:
            line.append(" ")
            line.append_text(self._progress_bar(task.progress, 8, color))
            line.append(f" {task.progress:.0%}", style="dim")

        # Duration
        if task.started_at:
            dur = task.duration()
            if dur:
                line.append(f" {self._format_duration(dur)}", style="dim")

        # Token count
        if task.tokens.total_tokens > 0:
            line.append(f" {self._format_tokens(task.tokens.total_tokens)}↔", style="dim #8B5CF6")

        # Cost
        if task.tokens.estimated_cost > 0:
            line.append(f" ${task.tokens.estimated_cost:.3f}", style="dim #10B981")

        # Agent
        if task.agent_id and task.agent_id in self.agents:
            agent = self.agents[task.agent_id]
            line.append(f" {agent.symbol}", style=agent.color)

        # Model
        if task.model != "default":
            model_short = task.model.split("-")[-1][:4]
            line.append(f" [{model_short}]", style="dim")

        # Dependency indicator
        if task.depends_on:
            blocked_count = len(self.registry.get_blocked_chain(task.id))
            if blocked_count > 0:
                line.append(f" ⏳{blocked_count}", style="#F59E0B")

        # Tool indicator
        if task.active_tool:
            line.append(f" ⚙{task.active_tool}", style="#F59E0B")

        # Error indicator
        if task.error:
            line.append(f" ⚠", style="#EF4444")

        lines.append(line)

        # Second line: streaming content or progress message
        if task.is_streaming and task.stream_buffer:
            content_line = Text()
            content_line.append("  " * depth + "    │ ", style="dim #8B5CF6")

            # Get last chunk
            last_chunk = task.stream_buffer[-1]
            content = last_chunk.content.replace("\n", " ")[:50]

            if last_chunk.chunk_type == ChunkType.THINKING:
                content_line.append(f"💭 {content}", style="italic #8B5CF6")
            elif last_chunk.chunk_type == ChunkType.TOOL_CALL:
                content_line.append(f"⚙ {content}", style="#F59E0B")
            else:
                content_line.append(content, style="dim")

            if last_chunk.token_count > 0:
                content_line.append(f" +{last_chunk.token_count}", style="dim")

            lines.append(content_line)

        # Third line: error message
        if task.error and task.state == TaskState.FAILED:
            error_line = Text()
            error_line.append("  " * depth + "    └─ ", style="dim")
            error_line.append(f"✗ {task.error[:50]}", style="italic #EF4444")
            lines.append(error_line)

        return lines

    def _render_tree_recursive(self, task_id: str, lines: List[Text], depth: int = 0):
        task = self.registry.get_task(task_id)
        if not task:
            return

        # Filter completed if needed
        if not self.show_completed and task.state == TaskState.SUCCESS:
            return

        # Filter subtasks if needed
        if not self.show_subtasks and task.task_type == "subtask":
            return

        task_lines = self._render_task_node(task, depth)
        lines.extend(task_lines)

        if not task.collapsed:
            for child_id in task.children:
                self._render_tree_recursive(child_id, lines, depth + 1)

    def _make_tree_panel(self) -> Panel:
        lines = []
        for root_id in self.registry.roots:
            self._render_tree_recursive(root_id, lines)

        if not lines:
            lines.append(Text("No active sagas. Waiting for tasks...", style="dim italic"))

        # Apply scroll
        visible = lines[self.tree_scroll:self.tree_scroll + 30]

        # Clear pulse effects
        self.pulse_tasks.clear()

        return Panel(
            Group(*visible),
            title=f"[bold]▼ Task Hierarchy ({self.registry.total_created} total, "
                  f"{len(self.registry.by_state[TaskState.RUNNING])} running)[/]",
            subtitle="[dim]↑↓ scroll • c collapse • h hide done • s subtasks[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Detail Panel Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_detail_panel(self) -> Panel:
        """Detailed view of selected task."""
        task = None
        if self.detail_task:
            task = self.registry.get_task(self.detail_task)

        # Find any active streaming task if none selected
        if not task:
            for t in self.registry.tasks.values():
                if t.is_streaming or t.state == TaskState.RUNNING:
                    task = t
                    break

        if not task:
            return Panel(
                Text("Select a task to view details", style="dim italic"),
                title="[bold]📋 Task Detail[/]",
                border_style="#374151",
                box=box.ROUNDED,
            )

        lines = []

        # Header
        cfg = STATE_CONFIG[task.state]
        header = Text()
        header.append(f"{cfg['symbol']} ", style=cfg["color"])
        header.append(f"{task.name}", style="bold")
        header.append(f" ({cfg['label']})", style=f"dim {cfg['color']}")
        lines.append(header)
        lines.append(Text("─" * 40, style="dim"))

        # Task info
        info = Text()
        info.append(f"ID: ", style="dim")
        info.append(f"{task.id}\n", style="")
        info.append(f"Type: ", style="dim")
        info.append(f"{task.task_type}\n", style="")
        if task.agent_id:
            agent = self.agents.get(task.agent_id)
            info.append(f"Agent: ", style="dim")
            if agent:
                info.append(f"{agent.symbol} {agent.name}\n", style=agent.color)
            else:
                info.append(f"{task.agent_id}\n", style="")
        info.append(f"Model: ", style="dim")
        info.append(f"{task.model}\n", style="")
        lines.append(info)

        # Timing
        timing = Text()
        timing.append("─ Timing ─\n", style="bold dim")
        if task.queued_at:
            wait = task.queue_wait_time()
            timing.append(f"Queue wait: {self._format_duration(wait or 0)}\n", style="dim")
        if task.started_at:
            timing.append(f"Duration: {self._format_duration(task.duration() or 0)}\n", style="")
        if task.latency.first_token_ms:
            timing.append(f"First token: {task.latency.first_token_ms:.0f}ms\n", style="dim")
        if task.latency.avg_chunk_latency > 0:
            timing.append(f"Avg latency: {task.latency.avg_chunk_latency:.0f}ms\n", style="dim")
        lines.append(timing)

        # Tokens & Cost
        tokens = Text()
        tokens.append("─ Tokens ─\n", style="bold dim")
        tokens.append(f"Input: {self._format_tokens(task.tokens.input_tokens)}\n", style="")
        tokens.append(f"Output: {self._format_tokens(task.tokens.output_tokens)}\n", style="")
        tokens.append(f"Total: {self._format_tokens(task.tokens.total_tokens)}\n", style="bold")
        tokens.append(f"Cost: ${task.tokens.estimated_cost:.4f}\n", style="#10B981")
        if task.latency.total_ms > 0 and task.tokens.output_tokens > 0:
            tps = task.tokens.output_tokens / (task.latency.total_ms / 1000)
            tokens.append(f"Speed: {tps:.1f} tok/s\n", style="dim")
        lines.append(tokens)

        # Tool calls
        if task.tool_calls:
            tools = Text()
            tools.append("─ Tool Calls ─\n", style="bold dim")
            for tc in task.tool_calls[-5:]:
                status = "✓" if tc.completed_at else "⏳"
                tools.append(f"{status} {tc.name}", style="#F59E0B")
                if tc.duration_ms > 0:
                    tools.append(f" ({tc.duration_ms:.0f}ms)", style="dim")
                tools.append("\n")
            lines.append(tools)

        # Stream buffer preview
        if task.stream_buffer:
            stream = Text()
            stream.append("─ Stream ─\n", style="bold dim")
            for chunk in list(task.stream_buffer)[-8:]:
                prefix = {"thinking": "💭", "tool_call": "⚙", "text": "│", "error": "✗"}.get(
                    chunk.chunk_type.value if isinstance(chunk.chunk_type, ChunkType) else chunk.chunk_type,
                    "·"
                )
                content = chunk.content.replace("\n", " ")[:35]
                stream.append(f"{prefix} {content}\n", style="dim")
            lines.append(stream)

        # Error
        if task.error:
            error = Text()
            error.append("─ Error ─\n", style="bold #EF4444")
            error.append(f"{task.error}\n", style="#EF4444")
            lines.append(error)

        # Result preview
        if task.result:
            result = Text()
            result.append("─ Result ─\n", style="bold #10B981")
            result.append(f"{task.result[:100]}...\n" if len(task.result) > 100 else f"{task.result}\n",
                         style="dim")
            lines.append(result)

        return Panel(
            Group(*lines),
            title=f"[bold]📋 {task.name[:20]}[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Activity Stream Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_stream_panel(self) -> Panel:
        lines = []
        events = self.stream.get_recent(25, min_severity=self.min_severity)

        for event in reversed(events):
            cfg = self.stream.type_config.get(event.event_type,
                                              {"icon": "·", "color": "#6B7280"})

            line = Text()
            line.append(f"{event.timestamp:7.1f}s ", style="dim")
            line.append(f"{cfg['icon']} ", style=cfg["color"])

            # Source
            task = self.registry.get_task(event.source_id)
            source_name = (task.name[:10] if task else event.source_id[:10])
            line.append(f"{source_name:<10} ", style="bold" if task else "dim")

            # Content
            content = event.content[:35]
            content_style = ""
            if event.severity == "error":
                content_style = "#EF4444"
            elif event.severity == "warning":
                content_style = "#F59E0B"
            line.append(content, style=content_style)

            # Detail
            if event.detail:
                line.append(f" │ {event.detail[:20]}", style="dim italic")

            lines.append(line)

        if not lines:
            lines.append(Text("Waiting for activity...", style="dim italic"))

        return Panel(
            Group(*lines),
            title=f"[bold]⚡ Activity ({len(self.stream.events)} events)[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Queue Panel Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_queue_panel(self) -> Panel:
        lines = []

        # Stats
        stats = Text()
        stats.append(f"Depth: {self.queue.size()}", style="bold")
        stats.append(f" │ Avg wait: {self.queue.avg_wait_time:.1f}s", style="dim")
        stats.append(f" │ Max: {self.queue.max_wait_time:.1f}s", style="dim")
        lines.append(stats)
        lines.append(Text(""))

        # Queued tasks
        queued = self.queue.peek(8)
        if queued:
            lines.append(Text("─ Pending ─", style="bold cyan"))
            for i, qt in enumerate(queued, 1):
                task = self.registry.get_task(qt.task_id)
                if not task:
                    continue

                line = Text()
                line.append(f"#{i:2d} ", style="dim")
                line.append(f"P{qt.priority} ", style="bold")

                wait = time.time() - qt.enqueued_at
                line.append(f"{task.name[:18]:<18}", style="")
                line.append(f" +{wait:.1f}s", style="#F59E0B")

                # Waiting on
                waiting = self.queue.waiting_on.get(qt.task_id, set())
                if waiting:
                    line.append(f" ⏳{len(waiting)}", style="dim")

                if qt.deadline:
                    remaining = qt.deadline - time.time()
                    if remaining > 0:
                        line.append(f" ⏱{remaining:.0f}s", style="#EC4899")
                    else:
                        line.append(" OVERDUE", style="bold #EF4444")

                lines.append(line)

        # Running
        running = list(self.registry.by_state[TaskState.RUNNING] |
                      self.registry.by_state[TaskState.STREAMING])
        if running:
            lines.append(Text(""))
            lines.append(Text("─ Running ─", style="bold #10B981"))
            for task_id in running[:5]:
                task = self.registry.get_task(task_id)
                if not task:
                    continue

                line = Text()
                spinner = SPINNERS[self.frame % len(SPINNERS)]
                line.append(f" {spinner} ", style="#10B981")
                line.append(f"{task.name[:18]:<18}", style="bold")

                if task.progress > 0:
                    line.append(" ")
                    line.append_text(self._progress_bar(task.progress, 6, "#10B981"))

                dur = task.duration()
                if dur:
                    line.append(f" {dur:.1f}s", style="dim")

                lines.append(line)

        # Blocked
        blocked = list(self.registry.by_state[TaskState.BLOCKED])[:3]
        if blocked:
            lines.append(Text(""))
            lines.append(Text("─ Blocked ─", style="bold #EF4444"))
            for task_id in blocked:
                task = self.registry.get_task(task_id)
                if not task:
                    continue

                line = Text()
                line.append(" ◈ ", style="#EF4444")
                line.append(f"{task.name[:18]:<18}", style="dim")

                chain = self.registry.get_blocked_chain(task_id)
                if chain:
                    blocker = self.registry.get_task(chain[0])
                    if blocker:
                        line.append(f" ← {blocker.name[:10]}", style="dim italic")

                lines.append(line)

        return Panel(
            Group(*lines),
            title="[bold]◌ Queue[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics Panel Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_metrics_panel(self) -> Panel:
        lines = []

        # Task stats
        task_stats = Text()
        task_stats.append("─ Tasks ─\n", style="bold")
        task_stats.append(f"Total: {self.metrics.total_tasks}\n", style="")
        task_stats.append(f"Completed: {self.metrics.completed_tasks}", style="#10B981")
        task_stats.append(f" ({self.metrics.completed_tasks/max(1,self.metrics.total_tasks)*100:.0f}%)\n", style="dim")
        task_stats.append(f"Failed: {self.metrics.failed_tasks}", style="#EF4444")
        if self.metrics.total_tasks > 0:
            err_rate = self.metrics.failed_tasks / self.metrics.total_tasks * 100
            task_stats.append(f" ({err_rate:.1f}%)\n", style="dim")
        task_stats.append(f"Active: {self.metrics.active_tasks}\n", style="#F59E0B")
        lines.append(task_stats)

        # Token stats
        token_stats = Text()
        token_stats.append("─ Tokens ─\n", style="bold")
        token_stats.append(f"Input: {self._format_tokens(self.metrics.total_input_tokens)}\n", style="")
        token_stats.append(f"Output: {self._format_tokens(self.metrics.total_output_tokens)}\n", style="")
        total_tok = self.metrics.total_input_tokens + self.metrics.total_output_tokens
        token_stats.append(f"Total: {self._format_tokens(total_tok)}\n", style="bold")
        lines.append(token_stats)

        # Cost
        cost_stats = Text()
        cost_stats.append("─ Cost ─\n", style="bold")
        cost_stats.append(f"Total: ${self.metrics.total_cost:.4f}\n", style="#10B981 bold")
        if self.elapsed() > 0:
            cost_per_min = self.metrics.total_cost / (self.elapsed() / 60)
            cost_stats.append(f"Rate: ${cost_per_min:.4f}/min\n", style="dim")
        lines.append(cost_stats)

        # Sparklines
        spark_lines = Text()
        spark_lines.append("─ Trends ─\n", style="bold")

        active_spark = self._sparkline(list(self.metrics.active_tasks_history))
        spark_lines.append(f"Active:  {active_spark}\n", style="#F59E0B")

        tps_spark = self._sparkline(list(self.metrics.tokens_per_second))
        spark_lines.append(f"Tok/s:   {tps_spark}\n", style="#8B5CF6")

        err_spark = self._sparkline(list(self.metrics.error_rate_history))
        spark_lines.append(f"Errors:  {err_spark}\n", style="#EF4444")

        lines.append(spark_lines)

        return Panel(
            Group(*lines),
            title="[bold]📊 Metrics[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Agents Panel Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_agents_panel(self) -> Panel:
        lines = []

        for agent in sorted(self.agents.values(), key=lambda a: a.id):
            line = Text()

            # State
            if agent.state == "running":
                line.append(SPINNERS[self.frame % len(SPINNERS)] + " ", style="#10B981")
            elif agent.state == "idle":
                line.append("○ ", style="dim")
            else:
                line.append("● ", style=agent.color)

            # Agent info
            line.append(f"{agent.symbol}", style=f"bold {agent.color}")
            line.append(f" {agent.name[:10]:<10}", style=agent.color)

            # Stats
            line.append(f" {agent.tasks_completed:3d}✓", style="dim #10B981")
            line.append(f" {agent.tasks_failed:2d}✗", style="dim #EF4444")
            line.append(f" {self._format_tokens(agent.total_tokens):>5}", style="dim")
            line.append(f" ${agent.total_cost:.2f}", style="dim #10B981")

            lines.append(line)

            # Current task
            if agent.current_task:
                task = self.registry.get_task(agent.current_task)
                if task:
                    task_line = Text()
                    task_line.append("    └─ ", style="dim")
                    task_line.append(f"{task.name[:25]}", style="italic")
                    if task.progress > 0:
                        task_line.append(f" {task.progress:.0%}", style="dim")
                    lines.append(task_line)

        if not lines:
            lines.append(Text("No agents registered", style="dim italic"))

        return Panel(
            Group(*lines),
            title=f"[bold]🤖 Agents ({len(self.agents)})[/]",
            border_style="#374151",
            box=box.ROUNDED,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Header Rendering
    # ─────────────────────────────────────────────────────────────────────────

    def _make_header(self) -> Text:
        spinner = SPINNERS[self.frame % len(SPINNERS)]

        header = Text()
        header.append(f" {spinner} ", style="bold cyan")
        header.append("LIDA", style="bold #7C3AED")
        header.append(" Hierarchical Dashboard ", style="bold white")
        header.append("│", style="dim")

        # Time
        header.append(f" ⏱{self.elapsed_str()}", style="#06B6D4")

        # Task counts
        running = len(self.registry.by_state[TaskState.RUNNING]) + len(self.registry.by_state[TaskState.STREAMING])
        header.append(f" ⚡{running}", style="#F59E0B")
        header.append(f" ✓{self.metrics.completed_tasks}", style="#10B981")
        header.append(f" ✗{self.metrics.failed_tasks}", style="#EF4444")

        # Tokens
        total_tok = self.metrics.total_input_tokens + self.metrics.total_output_tokens
        header.append(f" │ {self._format_tokens(total_tok)}tok", style="dim #8B5CF6")

        # Cost
        header.append(f" ${self.metrics.total_cost:.3f}", style="#10B981")

        # Queue
        header.append(f" │ Q:{self.queue.size()}", style="dim")

        header.append(" │", style="dim")

        # View tabs
        for mode in ViewMode:
            if mode == self.view_mode:
                header.append(f" [{mode.value.upper()}]", style="bold reverse #7C3AED")
            else:
                header.append(f" {mode.value}", style="dim")

        return header

    # ─────────────────────────────────────────────────────────────────────────
    # Main Layout
    # ─────────────────────────────────────────────────────────────────────────

    def make_layout(self) -> Layout:
        self.frame += 1

        # Update metrics
        running = len(self.registry.by_state[TaskState.RUNNING]) + len(self.registry.by_state[TaskState.STREAMING])
        self.metrics.active_tasks = running

        # Sample metrics periodically
        if self.frame % 10 == 0:
            tps = sum(t.tokens.output_tokens for t in self.registry.tasks.values()
                     if t.is_active()) / max(1, self.elapsed())
            err_rate = self.metrics.failed_tasks / max(1, self.metrics.total_tasks)
            self.metrics.record_sample(running, tps, err_rate, 0)

        layout = Layout()

        layout.split_column(
            Layout(name="header", size=1),
            Layout(name="body"),
            Layout(name="footer", size=1),
        )

        # Main body
        layout["body"].split_row(
            Layout(name="main", ratio=3),
            Layout(name="side", ratio=2),
        )

        # Main panel
        if self.view_mode == ViewMode.TREE:
            layout["main"].update(self._make_tree_panel())
        elif self.view_mode == ViewMode.DETAIL:
            layout["main"].split_column(
                Layout(name="tree", ratio=1),
                Layout(name="detail", ratio=1),
            )
            layout["tree"].update(self._make_tree_panel())
            layout["detail"].update(self._make_detail_panel())
        elif self.view_mode == ViewMode.METRICS:
            layout["main"].split_column(
                Layout(name="tree"),
                Layout(name="metrics", size=15),
            )
            layout["tree"].update(self._make_tree_panel())
            layout["metrics"].update(self._make_metrics_panel())
        else:
            layout["main"].update(self._make_tree_panel())

        # Side panels
        layout["side"].split_column(
            Layout(name="stream"),
            Layout(name="queue", size=14),
            Layout(name="agents", size=10),
        )

        layout["stream"].update(self._make_stream_panel())
        layout["queue"].update(self._make_queue_panel())
        layout["agents"].update(self._make_agents_panel())

        # Header & Footer
        layout["header"].update(Panel(self._make_header(), box=box.SIMPLE, padding=0))

        footer = Text()
        footer.append(" [1-5] views ", style="dim")
        footer.append("│ [c]ollapse [h]ide done [s]ubtasks ", style="dim")
        footer.append("│ [d]etail ", style="dim")
        footer.append("│ Ctrl+C quit ", style="dim")
        footer.append(f"│ {datetime.now().strftime('%H:%M:%S')} ", style="dim")
        layout["footer"].update(Panel(footer, box=box.SIMPLE, padding=0))

        return layout


# ═══════════════════════════════════════════════════════════════════════════════
# REALISTIC DEMO SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

class RealisticSimulator:
    """Simulates realistic LLM agent task execution."""

    def __init__(self, dashboard: HierarchicalDashboard):
        self.dashboard = dashboard
        self.thinking_phrases = [
            "Analyzing the requirements...",
            "Considering different approaches...",
            "Evaluating trade-offs...",
            "Reviewing existing patterns...",
            "Planning implementation strategy...",
            "Checking for edge cases...",
            "Validating assumptions...",
        ]
        self.tool_names = ["read_file", "search_code", "write_file", "run_tests",
                          "execute_bash", "web_search", "analyze_code"]

    async def setup_agents(self):
        agents = [
            ("agent-planner", "Planner", "planning", "#7C3AED", "◆",
             {"planning", "analysis"}, ["claude-3-opus", "claude-3-sonnet"]),
            ("agent-coder", "Coder", "coding", "#10B981", "●",
             {"coding", "refactoring"}, ["claude-3-sonnet", "claude-3-haiku"]),
            ("agent-reviewer", "Reviewer", "review", "#F59E0B", "★",
             {"review", "testing"}, ["claude-3-sonnet"]),
            ("agent-researcher", "Researcher", "research", "#06B6D4", "◇",
             {"research", "search"}, ["claude-3-opus"]),
        ]
        for aid, name, atype, color, symbol, caps, models in agents:
            self.dashboard.add_agent(aid, name, atype, color, symbol, caps, models)

    async def simulate_llm_stream(self, task_id: str, duration: float, tokens: int):
        """Simulate realistic LLM streaming."""
        task = self.dashboard.registry.get_task(task_id)
        if not task:
            return

        chunks_count = random.randint(15, 30)
        tokens_per_chunk = tokens // chunks_count
        chunk_delay = duration / chunks_count

        # Initial thinking
        for _ in range(random.randint(2, 4)):
            thought = random.choice(self.thinking_phrases)
            self.dashboard.stream_to_task(
                task_id, thought, ChunkType.THINKING,
                token_count=len(thought.split()),
                latency_ms=random.uniform(50, 150)
            )
            self.dashboard.update_progress(task_id, random.uniform(0.05, 0.15), "Thinking...")
            await asyncio.sleep(random.uniform(0.1, 0.3))

        # Maybe use a tool
        if random.random() > 0.5:
            tool_name = random.choice(self.tool_names)
            tool_id = self.dashboard.add_tool_call(
                task_id, tool_name,
                {"path": "/src/example.py", "query": "function definition"}
            )
            await asyncio.sleep(random.uniform(0.3, 0.8))
            self.dashboard.complete_tool_call(
                task_id, tool_id,
                result=f"Found 3 matches in {random.randint(1,5)} files"
            )

        # Main content generation
        for i in range(chunks_count):
            progress = 0.2 + (i / chunks_count) * 0.7

            content = f"{''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=random.randint(20, 60)))}"
            self.dashboard.stream_to_task(
                task_id, content, ChunkType.TEXT,
                token_count=tokens_per_chunk,
                latency_ms=random.uniform(30, 100)
            )
            self.dashboard.update_progress(task_id, progress)
            await asyncio.sleep(chunk_delay * random.uniform(0.8, 1.2))

    async def run_task(self, task_id: str, input_tokens: int = 500):
        """Run a single task with realistic timing."""
        task = self.dashboard.registry.get_task(task_id)
        if not task:
            return

        # Start
        self.dashboard.start_task(task_id, input_tokens)

        # Simulate streaming
        duration = random.uniform(2.0, 6.0)
        output_tokens = random.randint(200, 1000)

        await self.simulate_llm_stream(task_id, duration, output_tokens)

        # Complete or fail
        if random.random() > 0.1:
            result = f"Successfully completed analysis with {output_tokens} tokens generated."
            self.dashboard.complete_task(task_id, result, output_tokens)
        else:
            error = random.choice([
                "Rate limit exceeded",
                "Context length exceeded",
                "Invalid tool arguments",
                "Timeout waiting for response",
            ])
            should_retry = random.random() > 0.5
            self.dashboard.fail_task(task_id, error, retry=should_retry)

    async def run_saga(self, name: str, description: str = ""):
        """Run a complete saga with multiple tasks."""
        saga_id = self.dashboard.create_saga(name, description, {"demo"})

        # Create tasks
        num_tasks = random.randint(3, 6)
        task_ids = []
        prev_task = None

        for i in range(num_tasks):
            task_names = [
                "Analyze requirements",
                "Search codebase",
                "Plan implementation",
                "Write core logic",
                "Add error handling",
                "Write tests",
                "Review changes",
                "Generate documentation",
            ]
            task_name = random.choice(task_names)

            deps = [prev_task] if prev_task and random.random() > 0.3 else []
            agent = random.choice(list(self.dashboard.agents.keys()))

            task_id = self.dashboard.create_task(
                task_name,
                saga_id,
                depends_on=deps,
                agent_id=agent,
                priority=random.randint(0, 3),
                model=random.choice(["claude-3-sonnet", "claude-3-haiku"]),
                description=f"Task {i+1} of {num_tasks}",
                estimated_duration=random.uniform(2.0, 5.0),
            )

            # Add subtasks
            for j in range(random.randint(1, 3)):
                subtask_names = ["Parse input", "Validate data", "Transform", "Cache results"]
                self.dashboard.create_subtask(random.choice(subtask_names), task_id)

            task_ids.append(task_id)
            prev_task = task_id

        # Execute tasks
        for task_id in task_ids:
            task = self.dashboard.registry.get_task(task_id)
            if not task:
                continue

            # Wait for dependencies
            while True:
                deps_done = all(
                    self.dashboard.registry.get_task(d).state == TaskState.SUCCESS
                    for d in task.depends_on
                    if self.dashboard.registry.get_task(d)
                )
                deps_failed = any(
                    self.dashboard.registry.get_task(d).state == TaskState.FAILED
                    for d in task.depends_on
                    if self.dashboard.registry.get_task(d)
                )
                if deps_failed:
                    self.dashboard.fail_task(task_id, "Dependency failed")
                    break
                if deps_done:
                    break
                await asyncio.sleep(0.1)

            if task.state == TaskState.FAILED:
                continue

            await self.run_task(task_id)
            await asyncio.sleep(random.uniform(0.2, 0.5))

    async def run_continuous(self):
        """Run continuous realistic simulation."""
        await self.setup_agents()

        saga_templates = [
            ("Implement user authentication", "Add OAuth2 login flow"),
            ("Fix performance regression", "Optimize database queries"),
            ("Add new API endpoint", "Create REST API for user profiles"),
            ("Refactor legacy module", "Modernize authentication module"),
            ("Write integration tests", "Add E2E tests for checkout flow"),
            ("Update documentation", "Refresh API documentation"),
            ("Debug production issue", "Investigate memory leak"),
            ("Code review PR #247", "Review authentication changes"),
        ]

        while True:
            name, desc = random.choice(saga_templates)
            await self.run_saga(name, desc)
            await asyncio.sleep(random.uniform(1.0, 3.0))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    console.print("\n[bold #7C3AED]╔════════════════════════════════════════════════════════════════╗[/]")
    console.print("[bold #7C3AED]║[/]   [bold white]LIDA Hierarchical Real-Time Dashboard v2[/]                 [bold #7C3AED]║[/]")
    console.print("[bold #7C3AED]║[/]   [dim]Ultra-detailed task tracking with streaming & metrics[/]     [bold #7C3AED]║[/]")
    console.print("[bold #7C3AED]╚════════════════════════════════════════════════════════════════╝[/]\n")

    dashboard = HierarchicalDashboard()
    simulator = RealisticSimulator(dashboard)

    console.print("[cyan]Initializing hierarchical task system...[/]")
    await asyncio.sleep(0.5)
    console.print("[cyan]Setting up agent pool...[/]")
    await asyncio.sleep(0.5)
    console.print("[green]✓ Ready[/]")
    console.print("[dim]Starting dashboard...[/]\n")
    await asyncio.sleep(1)

    sim_task = asyncio.create_task(simulator.run_continuous())

    try:
        with Live(
            dashboard.make_layout(),
            console=console,
            refresh_per_second=12,
            screen=True,
        ) as live:
            while True:
                live.update(dashboard.make_layout())
                await asyncio.sleep(1/12)
    except KeyboardInterrupt:
        sim_task.cancel()
        console.print("\n[yellow]Dashboard stopped.[/]")


if __name__ == "__main__":
    asyncio.run(main())
