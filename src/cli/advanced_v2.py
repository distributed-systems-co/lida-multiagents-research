#!/usr/bin/env python3
"""
Advanced CLI features v2 - Enterprise-grade capabilities.

Provides:
- Real-time TUI dashboard
- Auto-scaling with load detection
- Circuit breakers and fault tolerance
- Distributed tracing
- Chaos engineering / fault injection
- Intelligent job scheduling
- State machine for service lifecycle
- Rate limiting
- Event bus for pub/sub
- Snapshot/restore system state
"""

from __future__ import annotations

import asyncio
import bisect
import collections
import contextlib
import functools
import hashlib
import heapq
import inspect
import json
import logging
import os
import queue
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Awaitable, Callable, Coroutine, Dict, Generic, Iterator,
    List, Optional, Protocol, Set, Tuple, Type, TypeVar, Union
)

T = TypeVar("T")
R = TypeVar("R")


# =============================================================================
# Event Bus - Pub/Sub System
# =============================================================================

@dataclass
class Event:
    """Base event class."""
    type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class EventBus:
    """In-process pub/sub event bus."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = defaultdict(list)
        self._async_subscribers: Dict[str, List[Callable[[Event], Awaitable[None]]]] = defaultdict(list)
        self._history: deque[Event] = deque(maxlen=1000)
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> Callable[[], None]:
        """Subscribe to an event type. Returns unsubscribe function."""
        with self._lock:
            self._subscribers[event_type].append(handler)

        def unsubscribe():
            with self._lock:
                self._subscribers[event_type].remove(handler)

        return unsubscribe

    def subscribe_async(self, event_type: str, handler: Callable[[Event], Awaitable[None]]) -> Callable[[], None]:
        """Subscribe async handler."""
        with self._lock:
            self._async_subscribers[event_type].append(handler)

        def unsubscribe():
            with self._lock:
                self._async_subscribers[event_type].remove(handler)

        return unsubscribe

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        with self._lock:
            self._history.append(event)
            handlers = list(self._subscribers.get(event.type, []))
            handlers.extend(self._subscribers.get("*", []))  # Wildcard subscribers

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logging.error(f"Event handler error: {e}")

    async def publish_async(self, event: Event) -> None:
        """Publish event and await async handlers."""
        self.publish(event)  # Sync handlers

        with self._lock:
            handlers = list(self._async_subscribers.get(event.type, []))
            handlers.extend(self._async_subscribers.get("*", []))

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logging.error(f"Async event handler error: {e}")

    def replay(self, event_type: Optional[str] = None, since: Optional[datetime] = None) -> List[Event]:
        """Replay events from history."""
        with self._lock:
            events = list(self._history)

        if event_type:
            events = [e for e in events if e.type == event_type]
        if since:
            events = [e for e in events if e.timestamp >= since]

        return events


# Global event bus instance
_event_bus = EventBus()


def get_event_bus() -> EventBus:
    return _event_bus


# =============================================================================
# Circuit Breaker - Fault Tolerance
# =============================================================================

class CircuitState(Enum):
    CLOSED = auto()      # Normal operation
    OPEN = auto()        # Failing, reject requests
    HALF_OPEN = auto()   # Testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_state_transition()
            return self._state

    def _check_state_transition(self) -> None:
        """Check if state should transition."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    self._success_count = 0

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        with self._lock:
            self._check_state_transition()

            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.OPEN:
                return False
            else:  # HALF_OPEN
                if self._half_open_calls < self.config.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap function with circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.allow_request():
                raise CircuitOpenError(f"Circuit {self.name} is OPEN")

            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                raise

        return wrapper

    def status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time,
            }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# Circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Get or create a circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


# =============================================================================
# Rate Limiter
# =============================================================================

class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens per second
            capacity: Maximum tokens (burst capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_update = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

    def acquire(self, tokens: float = 1.0, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket."""
        deadline = time.time() + timeout if timeout else None

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

            if not blocking:
                return False

            if deadline and time.time() >= deadline:
                return False

            # Wait for tokens to refill
            wait_time = (tokens - self._tokens) / self.rate
            time.sleep(min(wait_time, 0.1))

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to rate limit a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.acquire(blocking=True, timeout=30.0):
                raise RateLimitExceeded("Rate limit exceeded")
            return func(*args, **kwargs)

        return wrapper


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate limiting."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a request slot."""
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            # Remove old requests
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()

            if len(self._requests) < self.max_requests:
                self._requests.append(now)
                return True

            return False

    def remaining(self) -> int:
        """Get remaining requests in current window."""
        now = time.time()
        window_start = now - self.window_seconds

        with self._lock:
            while self._requests and self._requests[0] < window_start:
                self._requests.popleft()
            return max(0, self.max_requests - len(self._requests))


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    pass


# =============================================================================
# Distributed Tracing
# =============================================================================

@dataclass
class Span:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None

    def finish(self, status: str = "OK") -> None:
        self.end_time = time.time()
        self.status = status

    def log(self, message: str, **kwargs) -> None:
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            **kwargs
        })

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }


class Tracer:
    """Distributed tracing system."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._spans: Dict[str, List[Span]] = defaultdict(list)
        self._current_span: Optional[Span] = None
        self._lock = threading.Lock()
        self._exporters: List[Callable[[Span], None]] = []

    def start_span(
        self,
        operation_name: str,
        parent: Optional[Span] = None,
        trace_id: Optional[str] = None,
    ) -> Span:
        """Start a new span."""
        if trace_id is None:
            trace_id = parent.trace_id if parent else str(uuid.uuid4())[:16]

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:16],
            parent_span_id=parent.span_id if parent else None,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=time.time(),
        )

        with self._lock:
            self._spans[trace_id].append(span)
            self._current_span = span

        return span

    def finish_span(self, span: Span, status: str = "OK") -> None:
        """Finish a span and export it."""
        span.finish(status)

        for exporter in self._exporters:
            try:
                exporter(span)
            except Exception as e:
                logging.error(f"Trace exporter error: {e}")

    def add_exporter(self, exporter: Callable[[Span], None]) -> None:
        """Add a span exporter."""
        self._exporters.append(exporter)

    @contextlib.contextmanager
    def trace(self, operation_name: str, parent: Optional[Span] = None):
        """Context manager for tracing."""
        span = self.start_span(operation_name, parent)
        try:
            yield span
            self.finish_span(span, "OK")
        except Exception as e:
            span.set_tag("error", "true")
            span.log(f"Exception: {e}", level="error")
            self.finish_span(span, "ERROR")
            raise

    def get_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._lock:
            return list(self._spans.get(trace_id, []))

    def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """Export a trace as JSON."""
        spans = self.get_trace(trace_id)
        return {
            "trace_id": trace_id,
            "service": self.service_name,
            "spans": [s.to_dict() for s in spans],
            "span_count": len(spans),
        }


# Global tracer
_tracer: Optional[Tracer] = None


def get_tracer(service_name: str = "lida") -> Tracer:
    global _tracer
    if _tracer is None:
        _tracer = Tracer(service_name)
    return _tracer


# =============================================================================
# Auto-Scaler
# =============================================================================

@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown: float = 60.0
    scale_down_cooldown: float = 300.0
    scale_up_step: int = 2
    scale_down_step: int = 1


@dataclass
class ResourceMetrics:
    """Current resource usage metrics."""
    cpu_percent: float
    memory_percent: float
    request_rate: float  # requests per second
    response_time_p99: float  # 99th percentile in ms
    queue_depth: int
    timestamp: datetime = field(default_factory=datetime.now)


class AutoScaler:
    """Intelligent auto-scaling based on metrics."""

    def __init__(self, name: str, policy: Optional[ScalingPolicy] = None):
        self.name = name
        self.policy = policy or ScalingPolicy()
        self._current_instances = self.policy.min_instances
        self._last_scale_up: Optional[float] = None
        self._last_scale_down: Optional[float] = None
        self._metrics_history: deque[ResourceMetrics] = deque(maxlen=60)
        self._lock = threading.Lock()
        self._scale_callbacks: List[Callable[[int, int], None]] = []

    def record_metrics(self, metrics: ResourceMetrics) -> None:
        """Record current metrics."""
        with self._lock:
            self._metrics_history.append(metrics)

    def on_scale(self, callback: Callable[[int, int], None]) -> None:
        """Register callback for scaling events. Called with (old_count, new_count)."""
        self._scale_callbacks.append(callback)

    def _get_average_metrics(self, window: int = 5) -> Optional[ResourceMetrics]:
        """Get average metrics over recent window."""
        with self._lock:
            if len(self._metrics_history) < window:
                return None

            recent = list(self._metrics_history)[-window:]

        return ResourceMetrics(
            cpu_percent=statistics.mean(m.cpu_percent for m in recent),
            memory_percent=statistics.mean(m.memory_percent for m in recent),
            request_rate=statistics.mean(m.request_rate for m in recent),
            response_time_p99=statistics.mean(m.response_time_p99 for m in recent),
            queue_depth=int(statistics.mean(m.queue_depth for m in recent)),
        )

    def evaluate(self) -> Optional[int]:
        """Evaluate scaling decision. Returns new instance count or None."""
        metrics = self._get_average_metrics()
        if not metrics:
            return None

        now = time.time()
        policy = self.policy

        # Check for scale up
        should_scale_up = (
            metrics.cpu_percent > policy.scale_up_threshold or
            metrics.memory_percent > policy.scale_up_threshold or
            metrics.queue_depth > self._current_instances * 10
        )

        # Check for scale down
        should_scale_down = (
            metrics.cpu_percent < policy.scale_down_threshold and
            metrics.memory_percent < policy.scale_down_threshold and
            metrics.queue_depth < self._current_instances * 2
        )

        # Apply cooldowns
        if should_scale_up:
            if self._last_scale_up and (now - self._last_scale_up) < policy.scale_up_cooldown:
                return None

            new_count = min(
                self._current_instances + policy.scale_up_step,
                policy.max_instances
            )
            if new_count != self._current_instances:
                return new_count

        elif should_scale_down:
            if self._last_scale_down and (now - self._last_scale_down) < policy.scale_down_cooldown:
                return None

            new_count = max(
                self._current_instances - policy.scale_down_step,
                policy.min_instances
            )
            if new_count != self._current_instances:
                return new_count

        return None

    def apply_scaling(self, new_count: int) -> None:
        """Apply scaling decision."""
        old_count = self._current_instances

        with self._lock:
            self._current_instances = new_count
            if new_count > old_count:
                self._last_scale_up = time.time()
            else:
                self._last_scale_down = time.time()

        # Notify callbacks
        for callback in self._scale_callbacks:
            try:
                callback(old_count, new_count)
            except Exception as e:
                logging.error(f"Scale callback error: {e}")

        # Publish event
        get_event_bus().publish(Event(
            type="autoscaler.scaled",
            data={
                "name": self.name,
                "old_count": old_count,
                "new_count": new_count,
            }
        ))

    def status(self) -> Dict[str, Any]:
        """Get auto-scaler status."""
        metrics = self._get_average_metrics()
        return {
            "name": self.name,
            "current_instances": self._current_instances,
            "min_instances": self.policy.min_instances,
            "max_instances": self.policy.max_instances,
            "last_scale_up": self._last_scale_up,
            "last_scale_down": self._last_scale_down,
            "avg_cpu": metrics.cpu_percent if metrics else None,
            "avg_memory": metrics.memory_percent if metrics else None,
        }


# =============================================================================
# Chaos Engineering - Fault Injection
# =============================================================================

class FaultType(Enum):
    LATENCY = auto()
    ERROR = auto()
    TIMEOUT = auto()
    RESOURCE_EXHAUSTION = auto()
    NETWORK_PARTITION = auto()


@dataclass
class FaultConfig:
    """Fault injection configuration."""
    fault_type: FaultType
    probability: float  # 0.0 to 1.0
    duration: Optional[float] = None  # For latency
    error_message: str = "Injected fault"
    target_services: List[str] = field(default_factory=list)
    enabled: bool = True


class ChaosEngine:
    """Chaos engineering fault injection."""

    def __init__(self):
        self._faults: Dict[str, FaultConfig] = {}
        self._active = False
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = defaultdict(int)

    def register_fault(self, name: str, config: FaultConfig) -> None:
        """Register a fault configuration."""
        with self._lock:
            self._faults[name] = config

    def enable(self) -> None:
        """Enable chaos engineering."""
        self._active = True
        get_event_bus().publish(Event(
            type="chaos.enabled",
            data={"faults": list(self._faults.keys())}
        ))

    def disable(self) -> None:
        """Disable chaos engineering."""
        self._active = False
        get_event_bus().publish(Event(
            type="chaos.disabled",
            data={}
        ))

    def should_inject(self, fault_name: str, service: str = "") -> bool:
        """Check if fault should be injected."""
        if not self._active:
            return False

        with self._lock:
            config = self._faults.get(fault_name)
            if not config or not config.enabled:
                return False

            if config.target_services and service not in config.target_services:
                return False

            if random.random() < config.probability:
                self._stats[fault_name] += 1
                return True

            return False

    def inject(self, fault_name: str, service: str = "") -> None:
        """Inject a fault if conditions are met."""
        if not self.should_inject(fault_name, service):
            return

        with self._lock:
            config = self._faults.get(fault_name)
            if not config:
                return

        if config.fault_type == FaultType.LATENCY:
            time.sleep(config.duration or 1.0)

        elif config.fault_type == FaultType.ERROR:
            raise InjectedFaultError(config.error_message)

        elif config.fault_type == FaultType.TIMEOUT:
            time.sleep(config.duration or 30.0)
            raise TimeoutError(config.error_message)

        elif config.fault_type == FaultType.RESOURCE_EXHAUSTION:
            # Simulate memory pressure
            _ = [0] * (10 ** 7)  # ~80MB

    def __call__(self, fault_name: str):
        """Decorator to inject faults into functions."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                self.inject(fault_name)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    def status(self) -> Dict[str, Any]:
        """Get chaos engine status."""
        with self._lock:
            return {
                "active": self._active,
                "faults": {
                    name: {
                        "type": config.fault_type.name,
                        "probability": config.probability,
                        "enabled": config.enabled,
                        "injections": self._stats.get(name, 0),
                    }
                    for name, config in self._faults.items()
                }
            }


class InjectedFaultError(Exception):
    """Raised when a fault is injected."""
    pass


# Global chaos engine
_chaos_engine = ChaosEngine()


def get_chaos_engine() -> ChaosEngine:
    return _chaos_engine


# =============================================================================
# Job Scheduler
# =============================================================================

@dataclass
class ScheduledJob:
    """A scheduled job."""
    id: str
    name: str
    func: Callable[[], Any]
    schedule: str  # cron-like or interval
    next_run: datetime
    last_run: Optional[datetime] = None
    last_result: Optional[Any] = None
    last_error: Optional[str] = None
    enabled: bool = True
    max_retries: int = 3
    retry_count: int = 0


class Scheduler:
    """Job scheduler with cron-like scheduling."""

    def __init__(self):
        self._jobs: Dict[str, ScheduledJob] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def add_job(
        self,
        name: str,
        func: Callable[[], Any],
        interval: Optional[float] = None,
        cron: Optional[str] = None,
    ) -> str:
        """Add a scheduled job."""
        job_id = str(uuid.uuid4())[:8]

        if interval:
            schedule = f"every {interval}s"
            next_run = datetime.now() + timedelta(seconds=interval)
        elif cron:
            schedule = cron
            next_run = self._parse_cron_next(cron)
        else:
            raise ValueError("Must specify interval or cron")

        job = ScheduledJob(
            id=job_id,
            name=name,
            func=func,
            schedule=schedule,
            next_run=next_run,
        )

        with self._lock:
            self._jobs[job_id] = job

        return job_id

    def _parse_cron_next(self, cron: str) -> datetime:
        """Parse cron expression and get next run time."""
        # Simplified: just use interval for now
        # In production, use croniter or similar
        return datetime.now() + timedelta(minutes=1)

    def remove_job(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                return True
            return False

    def _run_job(self, job: ScheduledJob) -> None:
        """Execute a job."""
        try:
            result = job.func()
            job.last_result = result
            job.last_error = None
            job.retry_count = 0

            get_event_bus().publish(Event(
                type="scheduler.job_completed",
                data={"job_id": job.id, "name": job.name}
            ))

        except Exception as e:
            job.last_error = str(e)
            job.retry_count += 1

            get_event_bus().publish(Event(
                type="scheduler.job_failed",
                data={"job_id": job.id, "name": job.name, "error": str(e)}
            ))

        job.last_run = datetime.now()

        # Calculate next run
        if job.schedule.startswith("every "):
            interval = float(job.schedule.split()[1].rstrip("s"))
            job.next_run = datetime.now() + timedelta(seconds=interval)
        else:
            job.next_run = self._parse_cron_next(job.schedule)

    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now()

            with self._lock:
                jobs_to_run = [
                    job for job in self._jobs.values()
                    if job.enabled and job.next_run <= now
                ]

            for job in jobs_to_run:
                threading.Thread(target=self._run_job, args=(job,), daemon=True).start()

            time.sleep(1)

    def start(self) -> None:
        """Start the scheduler."""
        self._running = True
        self._thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        with self._lock:
            return {
                "running": self._running,
                "jobs": [
                    {
                        "id": job.id,
                        "name": job.name,
                        "schedule": job.schedule,
                        "next_run": job.next_run.isoformat(),
                        "last_run": job.last_run.isoformat() if job.last_run else None,
                        "enabled": job.enabled,
                    }
                    for job in self._jobs.values()
                ]
            }


# =============================================================================
# State Machine for Service Lifecycle
# =============================================================================

class StateTransition:
    """A state transition."""

    def __init__(
        self,
        from_state: str,
        to_state: str,
        trigger: str,
        guard: Optional[Callable[[], bool]] = None,
        action: Optional[Callable[[], None]] = None,
    ):
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger
        self.guard = guard
        self.action = action


class StateMachine:
    """State machine for managing lifecycle states."""

    def __init__(self, name: str, initial_state: str):
        self.name = name
        self._state = initial_state
        self._transitions: Dict[Tuple[str, str], StateTransition] = {}
        self._on_enter: Dict[str, List[Callable[[], None]]] = defaultdict(list)
        self._on_exit: Dict[str, List[Callable[[], None]]] = defaultdict(list)
        self._history: List[Tuple[datetime, str, str]] = []
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def add_transition(self, transition: StateTransition) -> None:
        """Add a state transition."""
        key = (transition.from_state, transition.trigger)
        self._transitions[key] = transition

    def on_enter(self, state: str, callback: Callable[[], None]) -> None:
        """Register callback for entering a state."""
        self._on_enter[state].append(callback)

    def on_exit(self, state: str, callback: Callable[[], None]) -> None:
        """Register callback for exiting a state."""
        self._on_exit[state].append(callback)

    def trigger(self, event: str) -> bool:
        """Trigger a state transition."""
        with self._lock:
            key = (self._state, event)
            transition = self._transitions.get(key)

            if not transition:
                return False

            # Check guard
            if transition.guard and not transition.guard():
                return False

            old_state = self._state

            # Exit callbacks
            for callback in self._on_exit.get(old_state, []):
                callback()

            # Transition action
            if transition.action:
                transition.action()

            # Update state
            self._state = transition.to_state
            self._history.append((datetime.now(), old_state, self._state))

            # Enter callbacks
            for callback in self._on_enter.get(self._state, []):
                callback()

            # Publish event
            get_event_bus().publish(Event(
                type="state_machine.transition",
                data={
                    "name": self.name,
                    "from_state": old_state,
                    "to_state": self._state,
                    "trigger": event,
                }
            ))

            return True

    def can_trigger(self, event: str) -> bool:
        """Check if an event can be triggered."""
        with self._lock:
            key = (self._state, event)
            transition = self._transitions.get(key)
            if not transition:
                return False
            if transition.guard and not transition.guard():
                return False
            return True

    def get_available_triggers(self) -> List[str]:
        """Get available triggers from current state."""
        with self._lock:
            return [
                trigger for (state, trigger), _ in self._transitions.items()
                if state == self._state
            ]


def create_service_state_machine(name: str) -> StateMachine:
    """Create a standard service lifecycle state machine."""
    sm = StateMachine(name, "stopped")

    # Define transitions
    sm.add_transition(StateTransition("stopped", "starting", "start"))
    sm.add_transition(StateTransition("starting", "running", "started"))
    sm.add_transition(StateTransition("starting", "failed", "error"))
    sm.add_transition(StateTransition("running", "stopping", "stop"))
    sm.add_transition(StateTransition("running", "degraded", "degrade"))
    sm.add_transition(StateTransition("running", "failed", "error"))
    sm.add_transition(StateTransition("degraded", "running", "recover"))
    sm.add_transition(StateTransition("degraded", "stopping", "stop"))
    sm.add_transition(StateTransition("degraded", "failed", "error"))
    sm.add_transition(StateTransition("stopping", "stopped", "stopped"))
    sm.add_transition(StateTransition("failed", "starting", "restart"))
    sm.add_transition(StateTransition("failed", "stopped", "reset"))

    return sm


# =============================================================================
# Snapshot & Restore
# =============================================================================

@dataclass
class Snapshot:
    """System state snapshot."""
    id: str
    name: str
    timestamp: datetime
    version: str
    data: Dict[str, Any]
    metadata: Dict[str, str] = field(default_factory=dict)


class SnapshotManager:
    """Manage system state snapshots."""

    def __init__(self, snapshot_dir: Optional[Path] = None):
        self.snapshot_dir = snapshot_dir or Path.home() / ".lida" / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        name: str,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, str]] = None,
    ) -> Snapshot:
        """Create a snapshot."""
        snapshot = Snapshot(
            id=str(uuid.uuid4())[:8],
            name=name,
            timestamp=datetime.now(),
            version="1.0",
            data=state,
            metadata=metadata or {},
        )

        # Save to file
        filename = f"{snapshot.id}_{name}_{snapshot.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.snapshot_dir / filename

        with open(filepath, "w") as f:
            json.dump({
                "id": snapshot.id,
                "name": snapshot.name,
                "timestamp": snapshot.timestamp.isoformat(),
                "version": snapshot.version,
                "data": snapshot.data,
                "metadata": snapshot.metadata,
            }, f, indent=2, default=str)

        get_event_bus().publish(Event(
            type="snapshot.created",
            data={"id": snapshot.id, "name": name}
        ))

        return snapshot

    def list(self) -> List[Snapshot]:
        """List all snapshots."""
        snapshots = []

        for filepath in self.snapshot_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                snapshots.append(Snapshot(
                    id=data["id"],
                    name=data["name"],
                    timestamp=datetime.fromisoformat(data["timestamp"]),
                    version=data["version"],
                    data=data["data"],
                    metadata=data.get("metadata", {}),
                ))
            except Exception:
                pass

        return sorted(snapshots, key=lambda s: s.timestamp, reverse=True)

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        """Get a snapshot by ID."""
        for filepath in self.snapshot_dir.glob(f"{snapshot_id}_*.json"):
            with open(filepath) as f:
                data = json.load(f)
            return Snapshot(
                id=data["id"],
                name=data["name"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                version=data["version"],
                data=data["data"],
                metadata=data.get("metadata", {}),
            )
        return None

    def restore(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Restore state from a snapshot."""
        snapshot = self.get(snapshot_id)
        if not snapshot:
            return None

        get_event_bus().publish(Event(
            type="snapshot.restored",
            data={"id": snapshot_id, "name": snapshot.name}
        ))

        return snapshot.data

    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot."""
        for filepath in self.snapshot_dir.glob(f"{snapshot_id}_*.json"):
            filepath.unlink()
            return True
        return False


# =============================================================================
# TUI Dashboard Components
# =============================================================================

class TerminalDashboard:
    """Real-time terminal dashboard."""

    def __init__(self):
        self._running = False
        self._panels: List[Callable[[], str]] = []
        self._refresh_rate = 1.0
        self._lock = threading.Lock()

    def add_panel(self, renderer: Callable[[], str]) -> None:
        """Add a panel renderer."""
        self._panels.append(renderer)

    def _clear_screen(self) -> None:
        """Clear terminal screen."""
        print("\033[2J\033[H", end="")

    def _render(self) -> str:
        """Render all panels."""
        lines = []
        lines.append("=" * 80)
        lines.append(" LIDA Dashboard".center(80))
        lines.append(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        lines.append("=" * 80)
        lines.append("")

        for panel in self._panels:
            try:
                content = panel()
                lines.append(content)
                lines.append("")
            except Exception as e:
                lines.append(f"[Panel Error: {e}]")
                lines.append("")

        lines.append("-" * 80)
        lines.append(" Press Ctrl+C to exit")

        return "\n".join(lines)

    def run(self) -> None:
        """Run the dashboard."""
        self._running = True

        try:
            while self._running:
                self._clear_screen()
                print(self._render())
                time.sleep(self._refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        """Stop the dashboard."""
        self._running = False


def create_services_panel(orchestrator) -> Callable[[], str]:
    """Create a services status panel."""
    def render() -> str:
        lines = ["┌─ Services " + "─" * 68 + "┐"]
        status = orchestrator.status()

        for name, info in status.items():
            state = info.get("state", "UNKNOWN")
            healthy = info.get("healthy", False)
            marker = "●" if healthy else "○"
            pid = info.get("pid", "-")
            lines.append(f"│ {marker} {name:15s} │ {state:12s} │ PID: {str(pid):8s} │")

        lines.append("└" + "─" * 78 + "┘")
        return "\n".join(lines)

    return render


def create_metrics_panel(collector) -> Callable[[], str]:
    """Create a metrics panel."""
    def render() -> str:
        lines = ["┌─ Metrics " + "─" * 69 + "┐"]

        for name in ["requests_total", "response_time_ms", "error_rate"]:
            latest = collector.get_latest(name)
            if latest:
                lines.append(f"│ {name:25s} │ {latest.value:>15.2f} │")

        lines.append("└" + "─" * 78 + "┘")
        return "\n".join(lines)

    return render


def create_events_panel(event_bus: EventBus, max_events: int = 5) -> Callable[[], str]:
    """Create a recent events panel."""
    def render() -> str:
        lines = ["┌─ Recent Events " + "─" * 62 + "┐"]

        events = event_bus.replay()[-max_events:]
        for event in reversed(events):
            ts = event.timestamp.strftime("%H:%M:%S")
            lines.append(f"│ [{ts}] {event.type:30s} │ {str(event.data)[:30]:30s} │")

        if not events:
            lines.append("│ No recent events".ljust(78) + "│")

        lines.append("└" + "─" * 78 + "┘")
        return "\n".join(lines)

    return render


# =============================================================================
# Health Check Protocol
# =============================================================================

class HealthStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class HealthChecker:
    """Comprehensive health checking system."""

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}

    def register(self, name: str, check: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        self._checks[name] = check

    def check(self, name: Optional[str] = None) -> Dict[str, HealthCheckResult]:
        """Run health checks."""
        if name:
            checks = {name: self._checks[name]} if name in self._checks else {}
        else:
            checks = self._checks

        results = {}
        for check_name, check_func in checks.items():
            try:
                results[check_name] = check_func()
            except Exception as e:
                results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e),
                )

        return results

    def is_healthy(self) -> bool:
        """Check if all checks pass."""
        results = self.check()
        return all(r.status == HealthStatus.HEALTHY for r in results.values())

    def summary(self) -> Dict[str, Any]:
        """Get health summary."""
        results = self.check()
        return {
            "status": "healthy" if self.is_healthy() else "unhealthy",
            "checks": {
                name: {
                    "status": result.status.name,
                    "message": result.message,
                }
                for name, result in results.items()
            }
        }


# Common health checks
def redis_health_check(port: int = 6379) -> HealthCheckResult:
    """Check Redis connectivity."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            result = s.connect_ex(('localhost', port))
            if result == 0:
                return HealthCheckResult(
                    name="redis",
                    status=HealthStatus.HEALTHY,
                    message=f"Redis responding on port {port}",
                )
            return HealthCheckResult(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis not responding on port {port}",
            )
    except Exception as e:
        return HealthCheckResult(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


def api_health_check(port: int = 2040) -> HealthCheckResult:
    """Check API server health."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as resp:
            if resp.status == 200:
                return HealthCheckResult(
                    name="api",
                    status=HealthStatus.HEALTHY,
                    message=f"API healthy on port {port}",
                )
    except Exception as e:
        pass

    return HealthCheckResult(
        name="api",
        status=HealthStatus.UNHEALTHY,
        message=f"API not responding on port {port}",
    )


def disk_health_check(path: str = "/", threshold: float = 90.0) -> HealthCheckResult:
    """Check disk space."""
    try:
        usage = shutil.disk_usage(path)
        percent = (usage.used / usage.total) * 100

        if percent < threshold:
            return HealthCheckResult(
                name="disk",
                status=HealthStatus.HEALTHY,
                message=f"Disk usage: {percent:.1f}%",
                details={"percent": percent, "free_gb": usage.free / (1024**3)},
            )
        return HealthCheckResult(
            name="disk",
            status=HealthStatus.DEGRADED,
            message=f"Disk usage high: {percent:.1f}%",
            details={"percent": percent},
        )
    except Exception as e:
        return HealthCheckResult(
            name="disk",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
        )


# =============================================================================
# Retry Policies
# =============================================================================

@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


def retry(policy: Optional[RetryPolicy] = None):
    """Decorator for automatic retries with exponential backoff."""
    if policy is None:
        policy = RetryPolicy()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(policy.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except policy.retryable_exceptions as e:
                    last_exception = e

                    if attempt == policy.max_retries:
                        break

                    # Calculate delay
                    delay = min(
                        policy.base_delay * (policy.exponential_base ** attempt),
                        policy.max_delay
                    )

                    if policy.jitter:
                        delay *= (0.5 + random.random())

                    time.sleep(delay)

            raise last_exception

        return wrapper

    return decorator


async def retry_async(
    func: Callable[..., Awaitable[T]],
    policy: Optional[RetryPolicy] = None,
    *args,
    **kwargs
) -> T:
    """Async retry with exponential backoff."""
    if policy is None:
        policy = RetryPolicy()

    last_exception = None

    for attempt in range(policy.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except policy.retryable_exceptions as e:
            last_exception = e

            if attempt == policy.max_retries:
                break

            delay = min(
                policy.base_delay * (policy.exponential_base ** attempt),
                policy.max_delay
            )

            if policy.jitter:
                delay *= (0.5 + random.random())

            await asyncio.sleep(delay)

    raise last_exception
