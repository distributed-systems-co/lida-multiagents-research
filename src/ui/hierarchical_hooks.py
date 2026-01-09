"""
Hierarchical Dashboard Hooks

Integration layer to connect real agent execution to the hierarchical dashboard.
Provides decorators and context managers for automatic task tracking.
"""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar, ParamSpec
from collections import deque
import threading

# Global dashboard instance (set by the dashboard runner)
_dashboard = None
_lock = threading.Lock()


def set_dashboard(dashboard):
    """Set the global dashboard instance."""
    global _dashboard
    with _lock:
        _dashboard = dashboard


def get_dashboard():
    """Get the global dashboard instance."""
    with _lock:
        return _dashboard


# ═══════════════════════════════════════════════════════════════════════════════
# TASK CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskContext:
    """Context for a running task."""
    task_id: str
    saga_id: Optional[str] = None
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    _stream_buffer: deque = field(default_factory=lambda: deque(maxlen=100))


# Thread-local storage for task context stack
_context_stack = threading.local()


def _get_context_stack() -> List[TaskContext]:
    if not hasattr(_context_stack, 'stack'):
        _context_stack.stack = []
    return _context_stack.stack


def current_task() -> Optional[TaskContext]:
    """Get the current task context."""
    stack = _get_context_stack()
    return stack[-1] if stack else None


def current_saga() -> Optional[str]:
    """Get the current saga ID."""
    ctx = current_task()
    return ctx.saga_id if ctx else None


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT MANAGERS
# ═══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def saga(name: str, description: str = "", tags: Set[str] = None):
    """
    Context manager for a saga (top-level operation).

    Usage:
        async with saga("Analyze codebase") as s:
            async with task("Read files"):
                ...
    """
    dashboard = get_dashboard()
    saga_id = None

    try:
        if dashboard:
            saga_id = dashboard.create_saga(name, description, tags)
            dashboard.start_task(saga_id)

        ctx = TaskContext(
            task_id=saga_id or f"saga-{uuid.uuid4().hex[:8]}",
            saga_id=saga_id,
        )
        _get_context_stack().append(ctx)

        yield ctx

        if dashboard and saga_id:
            dashboard.complete_task(saga_id, "Saga completed")

    except Exception as e:
        if dashboard and saga_id:
            dashboard.fail_task(saga_id, str(e))
        raise
    finally:
        stack = _get_context_stack()
        if stack:
            stack.pop()


@asynccontextmanager
async def task(name: str, depends_on: List[str] = None, agent_id: str = None,
               priority: int = 0, tags: Set[str] = None):
    """
    Context manager for a task within a saga.

    Usage:
        async with task("Process data", depends_on=["task-1"]):
            result = await process()
    """
    dashboard = get_dashboard()
    parent_ctx = current_task()
    task_id = None

    try:
        parent_id = parent_ctx.task_id if parent_ctx else None
        saga_id = parent_ctx.saga_id if parent_ctx else None

        if dashboard and parent_id:
            task_id = dashboard.create_task(
                name,
                parent_id,
                depends_on=depends_on,
                agent_id=agent_id,
                priority=priority,
                tags=tags,
            )
            dashboard.start_task(task_id)

        ctx = TaskContext(
            task_id=task_id or f"task-{uuid.uuid4().hex[:8]}",
            saga_id=saga_id,
            parent_id=parent_id,
        )
        _get_context_stack().append(ctx)

        yield ctx

        if dashboard and task_id:
            dashboard.complete_task(task_id)

    except Exception as e:
        if dashboard and task_id:
            dashboard.fail_task(task_id, str(e))
        raise
    finally:
        stack = _get_context_stack()
        if stack:
            stack.pop()


@asynccontextmanager
async def subtask(name: str):
    """Context manager for a subtask."""
    dashboard = get_dashboard()
    parent_ctx = current_task()
    subtask_id = None

    try:
        parent_id = parent_ctx.task_id if parent_ctx else None
        saga_id = parent_ctx.saga_id if parent_ctx else None

        if dashboard and parent_id:
            subtask_id = dashboard.create_subtask(name, parent_id)
            dashboard.start_task(subtask_id)

        ctx = TaskContext(
            task_id=subtask_id or f"sub-{uuid.uuid4().hex[:8]}",
            saga_id=saga_id,
            parent_id=parent_id,
        )
        _get_context_stack().append(ctx)

        yield ctx

        if dashboard and subtask_id:
            dashboard.complete_task(subtask_id)

    except Exception as e:
        if dashboard and subtask_id:
            dashboard.fail_task(subtask_id, str(e))
        raise
    finally:
        stack = _get_context_stack()
        if stack:
            stack.pop()


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING
# ═══════════════════════════════════════════════════════════════════════════════

def stream(content: str, chunk_type: str = "text"):
    """Stream content to the current task."""
    ctx = current_task()
    dashboard = get_dashboard()

    if ctx and dashboard:
        dashboard.stream_to_task(ctx.task_id, content, chunk_type)

    # Also store locally
    if ctx:
        ctx._stream_buffer.append((time.time(), chunk_type, content))


def update_progress(progress: float):
    """Update progress of the current task."""
    ctx = current_task()
    dashboard = get_dashboard()

    if ctx and dashboard:
        dashboard.update_progress(ctx.task_id, progress)


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

P = ParamSpec('P')
T = TypeVar('T')


def tracked_saga(name: str = None, description: str = "", tags: Set[str] = None):
    """
    Decorator to track a function as a saga.

    Usage:
        @tracked_saga("Analyze codebase")
        async def analyze():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        saga_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with saga(saga_name, description, tags):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def tracked_task(name: str = None, depends_on: List[str] = None,
                 agent_id: str = None, priority: int = 0):
    """
    Decorator to track a function as a task.

    Usage:
        @tracked_task("Process data")
        async def process():
            ...
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        task_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with task(task_name, depends_on, agent_id, priority):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


def tracked_subtask(name: str = None):
    """Decorator to track a function as a subtask."""
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        subtask_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with subtask(subtask_name):
                return await func(*args, **kwargs)

        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# LLM STREAMING ADAPTER
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingAdapter:
    """
    Adapter to capture LLM streaming responses and route to dashboard.

    Usage:
        adapter = StreamingAdapter()
        async for chunk in llm.stream(prompt):
            adapter.on_chunk(chunk)
            yield chunk
    """

    def __init__(self, task_context: TaskContext = None):
        self.context = task_context or current_task()
        self.total_tokens = 0
        self.chunks: List[str] = []

    def on_chunk(self, chunk: str, chunk_type: str = "text"):
        """Handle a streaming chunk."""
        self.chunks.append(chunk)
        self.total_tokens += len(chunk.split())
        stream(chunk, chunk_type)

    def on_tool_call(self, tool_name: str, args: Dict[str, Any]):
        """Handle a tool call."""
        stream(f"Calling {tool_name}...", "tool_call")

    def on_tool_result(self, tool_name: str, result: Any):
        """Handle a tool result."""
        stream(f"{tool_name} returned", "tool_result")

    def on_thought(self, thought: str):
        """Handle a thinking/reasoning chunk."""
        stream(thought, "thought")

    def on_error(self, error: str):
        """Handle an error."""
        stream(f"Error: {error}", "error")

    def get_full_response(self) -> str:
        """Get the full accumulated response."""
        return "".join(self.chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class AgentHook:
    """
    Hook to integrate agents with the dashboard.

    Usage:
        hook = AgentHook("agent-1", "Researcher", "research", "#06B6D4", "◆")
        await hook.execute_task("Analyze code", analyze_func)
    """

    def __init__(self, agent_id: str, name: str, agent_type: str,
                 color: str, symbol: str, capabilities: Set[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.agent_type = agent_type
        self.color = color
        self.symbol = symbol
        self.capabilities = capabilities or set()
        self._registered = False

    def register(self):
        """Register this agent with the dashboard."""
        dashboard = get_dashboard()
        if dashboard and not self._registered:
            dashboard.add_agent(
                self.agent_id,
                self.name,
                self.agent_type,
                self.color,
                self.symbol,
                self.capabilities
            )
            self._registered = True

    async def execute_task(self, name: str, func: Callable, *args, **kwargs):
        """Execute a task and track it."""
        self.register()

        async with task(name, agent_id=self.agent_id):
            return await func(*args, **kwargs)

    async def execute_subtask(self, name: str, func: Callable, *args, **kwargs):
        """Execute a subtask and track it."""
        async with subtask(name):
            return await func(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def emit_event(event_type: str, content: str, metadata: Dict[str, Any] = None):
    """Emit a custom event to the activity stream."""
    ctx = current_task()
    dashboard = get_dashboard()

    if dashboard:
        source_id = ctx.task_id if ctx else "system"
        dashboard.stream.emit(event_type, source_id, content, metadata=metadata)


def log_dependency(from_task: str, to_task: str, reason: str = ""):
    """Log a dependency relationship."""
    dashboard = get_dashboard()
    if dashboard:
        dashboard.stream.emit(
            "dependency_added",
            from_task,
            f"Depends on {to_task}: {reason}",
            target_id=to_task
        )


def mark_blocked(reason: str):
    """Mark the current task as blocked."""
    ctx = current_task()
    dashboard = get_dashboard()

    if ctx and dashboard:
        task_node = dashboard.registry.get_task(ctx.task_id)
        if task_node:
            from run_hierarchical_dashboard import TaskState
            task_node.state = TaskState.BLOCKED
            dashboard.stream.emit("task_blocked", ctx.task_id, f"Blocked: {reason}")


def add_tag(tag: str):
    """Add a tag to the current task."""
    ctx = current_task()
    dashboard = get_dashboard()

    if ctx and dashboard:
        task_node = dashboard.registry.get_task(ctx.task_id)
        if task_node:
            task_node.tags.add(tag)
