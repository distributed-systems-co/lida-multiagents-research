#!/usr/bin/env python3
"""
Advanced CLI features for LIDA.

Provides:
- Service orchestration with dependencies
- Configuration profiles
- Health monitoring with auto-restart
- Distributed coordination via Redis
- Pipeline execution
- Cluster management
- Real-time metrics
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar
from contextlib import contextmanager

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows compatibility


# =============================================================================
# Service Orchestration
# =============================================================================

class ServiceState(Enum):
    """Service lifecycle states."""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    DEGRADED = auto()
    STOPPING = auto()
    FAILED = auto()
    UNKNOWN = auto()


@dataclass
class ServiceConfig:
    """Configuration for a managed service."""
    name: str
    command: List[str]
    port: Optional[int] = None
    health_endpoint: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    restart_policy: str = "on-failure"  # always, on-failure, never
    restart_delay: float = 2.0
    max_restarts: int = 5
    startup_timeout: float = 30.0
    health_check_interval: float = 10.0
    working_dir: Optional[str] = None


@dataclass
class ServiceInstance:
    """Runtime state of a service instance."""
    config: ServiceConfig
    state: ServiceState = ServiceState.STOPPED
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    restart_count: int = 0
    last_health_check: Optional[datetime] = None
    last_error: Optional[str] = None


class ServiceOrchestrator:
    """Manages service lifecycle with dependencies."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.services: Dict[str, ServiceInstance] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

    def register(self, config: ServiceConfig) -> None:
        """Register a service configuration."""
        with self._lock:
            self.services[config.name] = ServiceInstance(config=config)

    def register_default_services(self, redis_port: int = 6379, api_port: int = 2040) -> None:
        """Register default LIDA services."""
        self.register(ServiceConfig(
            name="redis",
            command=["redis-server", "--port", str(redis_port)],
            port=redis_port,
            health_endpoint=None,  # Uses port check
            restart_policy="always",
        ))

        self.register(ServiceConfig(
            name="api",
            command=[sys.executable, str(self.project_root / "run_swarm_server.py"), f"--port={api_port}"],
            port=api_port,
            health_endpoint="/health",
            depends_on=["redis"],
            environment={"REDIS_URL": f"redis://localhost:{redis_port}"},
            restart_policy="on-failure",
        ))

        self.register(ServiceConfig(
            name="workers",
            command=[sys.executable, str(self.project_root / "run_workers.py"), "--num-workers", "4"],
            depends_on=["redis"],
            environment={"REDIS_URL": f"redis://localhost:{redis_port}"},
            restart_policy="on-failure",
        ))

    def _check_port(self, port: int) -> bool:
        """Check if a port is accepting connections."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                return s.connect_ex(('localhost', port)) == 0
        except Exception:
            return False

    def _check_health(self, instance: ServiceInstance) -> bool:
        """Check service health."""
        config = instance.config

        # Process check
        if instance.process and instance.process.poll() is not None:
            return False

        # Port check
        if config.port and not self._check_port(config.port):
            return False

        # HTTP health check
        if config.port and config.health_endpoint:
            try:
                import urllib.request
                url = f"http://localhost:{config.port}{config.health_endpoint}"
                with urllib.request.urlopen(url, timeout=5) as resp:
                    return resp.status == 200
            except Exception:
                return False

        return True

    def _resolve_dependencies(self, names: List[str]) -> List[str]:
        """Topologically sort services by dependencies."""
        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            if name in self.services:
                for dep in self.services[name].config.depends_on:
                    visit(dep)
                order.append(name)

        for name in names:
            visit(name)
        return order

    def _start_service(self, name: str) -> bool:
        """Start a single service."""
        instance = self.services.get(name)
        if not instance:
            return False

        config = instance.config

        # Check dependencies
        for dep in config.depends_on:
            dep_instance = self.services.get(dep)
            if not dep_instance or dep_instance.state != ServiceState.RUNNING:
                instance.last_error = f"Dependency {dep} not running"
                return False

        # Build environment
        env = os.environ.copy()
        env.update(config.environment)

        # Start process
        try:
            instance.state = ServiceState.STARTING
            instance.process = subprocess.Popen(
                config.command,
                cwd=config.working_dir or str(self.project_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            instance.pid = instance.process.pid
            instance.started_at = datetime.now()

            # Wait for startup
            deadline = time.time() + config.startup_timeout
            while time.time() < deadline:
                if self._check_health(instance):
                    instance.state = ServiceState.RUNNING
                    return True
                if instance.process.poll() is not None:
                    instance.state = ServiceState.FAILED
                    instance.last_error = "Process exited during startup"
                    return False
                time.sleep(0.5)

            instance.state = ServiceState.FAILED
            instance.last_error = "Startup timeout"
            return False

        except Exception as e:
            instance.state = ServiceState.FAILED
            instance.last_error = str(e)
            return False

    def _stop_service(self, name: str, timeout: float = 10.0) -> bool:
        """Stop a single service."""
        instance = self.services.get(name)
        if not instance or not instance.process:
            return True

        instance.state = ServiceState.STOPPING

        try:
            instance.process.terminate()
            try:
                instance.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                instance.process.kill()
                instance.process.wait(timeout=5)
        except Exception:
            pass

        instance.state = ServiceState.STOPPED
        instance.process = None
        instance.pid = None
        return True

    def start(self, names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Start services in dependency order."""
        if names is None:
            names = list(self.services.keys())

        order = self._resolve_dependencies(names)
        results = {}

        for name in order:
            results[name] = self._start_service(name)
            if not results[name]:
                break

        return results

    def stop(self, names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Stop services in reverse dependency order."""
        if names is None:
            names = list(self.services.keys())

        order = list(reversed(self._resolve_dependencies(names)))
        results = {}

        for name in order:
            results[name] = self._stop_service(name)

        return results

    def restart(self, name: str) -> bool:
        """Restart a service."""
        self._stop_service(name)
        time.sleep(self.services[name].config.restart_delay)
        return self._start_service(name)

    def status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        result = {}
        for name, instance in self.services.items():
            result[name] = {
                "state": instance.state.name,
                "pid": instance.pid,
                "started_at": instance.started_at.isoformat() if instance.started_at else None,
                "restart_count": instance.restart_count,
                "last_error": instance.last_error,
                "port": instance.config.port,
                "healthy": self._check_health(instance) if instance.state == ServiceState.RUNNING else False,
            }
        return result

    def start_monitor(self, callback: Optional[Callable[[str, ServiceState], None]] = None) -> None:
        """Start background health monitoring."""
        self._running = True

        def monitor_loop():
            while self._running:
                with self._lock:
                    for name, instance in self.services.items():
                        if instance.state == ServiceState.RUNNING:
                            if not self._check_health(instance):
                                instance.state = ServiceState.DEGRADED

                                if callback:
                                    callback(name, ServiceState.DEGRADED)

                                # Auto-restart
                                policy = instance.config.restart_policy
                                if policy in ("always", "on-failure"):
                                    if instance.restart_count < instance.config.max_restarts:
                                        instance.restart_count += 1
                                        self.restart(name)
                                        if callback:
                                            callback(name, instance.state)

                            instance.last_health_check = datetime.now()

                time.sleep(5)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitor(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)


# =============================================================================
# Configuration Profiles
# =============================================================================

@dataclass
class Profile:
    """Named configuration profile."""
    name: str
    redis_port: int = 6379
    api_port: int = 2040
    workers: int = 4
    live_mode: bool = False
    scenario: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)
    services: List[str] = field(default_factory=lambda: ["redis", "api", "workers"])
    created_at: Optional[str] = None
    description: str = ""


class ProfileManager:
    """Manage named configuration profiles."""

    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path.home() / ".config" / "lida"
        self.profiles_file = self.config_dir / "profiles.json"
        self._ensure_dir()
        self._profiles: Dict[str, Profile] = {}
        self._load()

    def _ensure_dir(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file) as f:
                    data = json.load(f)
                for name, pdata in data.items():
                    self._profiles[name] = Profile(**pdata)
            except Exception:
                pass

    def _save(self) -> None:
        data = {}
        for name, profile in self._profiles.items():
            data[name] = {
                "name": profile.name,
                "redis_port": profile.redis_port,
                "api_port": profile.api_port,
                "workers": profile.workers,
                "live_mode": profile.live_mode,
                "scenario": profile.scenario,
                "environment": profile.environment,
                "services": profile.services,
                "created_at": profile.created_at,
                "description": profile.description,
            }
        with open(self.profiles_file, "w") as f:
            json.dump(data, f, indent=2)

    def create(self, name: str, **kwargs) -> Profile:
        """Create a new profile."""
        kwargs["name"] = name
        kwargs["created_at"] = datetime.now().isoformat()
        profile = Profile(**kwargs)
        self._profiles[name] = profile
        self._save()
        return profile

    def get(self, name: str) -> Optional[Profile]:
        """Get a profile by name."""
        return self._profiles.get(name)

    def list(self) -> List[Profile]:
        """List all profiles."""
        return list(self._profiles.values())

    def delete(self, name: str) -> bool:
        """Delete a profile."""
        if name in self._profiles:
            del self._profiles[name]
            self._save()
            return True
        return False

    def set_default(self, name: str) -> bool:
        """Set the default profile."""
        if name in self._profiles:
            self._profiles["_default"] = self._profiles[name]
            self._save()
            return True
        return False

    def get_default(self) -> Optional[Profile]:
        """Get the default profile."""
        return self._profiles.get("_default")

    def export_env(self, name: str) -> str:
        """Export profile as shell environment."""
        profile = self.get(name)
        if not profile:
            return ""

        lines = [
            f"# LIDA Profile: {name}",
            f"export REDIS_PORT={profile.redis_port}",
            f"export REDIS_URL=redis://localhost:{profile.redis_port}",
            f"export API_PORT={profile.api_port}",
            f"export PORT={profile.api_port}",
            f"export LIDA_WORKERS={profile.workers}",
        ]
        if profile.live_mode:
            lines.append("export SWARM_LIVE=true")
        if profile.scenario:
            lines.append(f"export SCENARIO={profile.scenario}")
        for k, v in profile.environment.items():
            lines.append(f"export {k}={v}")

        return "\n".join(lines)


# =============================================================================
# Distributed Coordination
# =============================================================================

class DistributedLock:
    """Redis-based distributed lock for cluster coordination."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None

    def _get_redis(self):
        if self._redis is None:
            try:
                import redis
                self._redis = redis.from_url(self.redis_url)
            except ImportError:
                raise RuntimeError("redis package required: pip install redis")
        return self._redis

    @contextmanager
    def acquire(self, name: str, timeout: float = 30.0, blocking: bool = True):
        """Acquire a distributed lock."""
        r = self._get_redis()
        lock_key = f"lida:lock:{name}"
        lock_id = f"{socket.gethostname()}:{os.getpid()}:{time.time()}"

        acquired = False
        start = time.time()

        while True:
            # Try to acquire
            if r.set(lock_key, lock_id, nx=True, ex=int(timeout)):
                acquired = True
                break

            if not blocking:
                break

            if time.time() - start > timeout:
                break

            time.sleep(0.1)

        try:
            if acquired:
                yield True
            else:
                yield False
        finally:
            if acquired:
                # Only release if we still hold it
                if r.get(lock_key) == lock_id.encode():
                    r.delete(lock_key)

    def is_locked(self, name: str) -> bool:
        """Check if a lock is held."""
        r = self._get_redis()
        return r.exists(f"lida:lock:{name}") > 0

    def force_release(self, name: str) -> bool:
        """Force release a lock (admin operation)."""
        r = self._get_redis()
        return r.delete(f"lida:lock:{name}") > 0


class ServiceRegistry:
    """Redis-based service registry for discovery."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self._redis = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

    def _get_redis(self):
        if self._redis is None:
            import redis
            self._redis = redis.from_url(self.redis_url)
        return self._redis

    def register(self, service_name: str, host: str, port: int, metadata: Dict[str, Any] = None) -> str:
        """Register a service instance."""
        r = self._get_redis()
        instance_id = f"{host}:{port}"
        key = f"lida:service:{service_name}:{instance_id}"

        data = {
            "host": host,
            "port": port,
            "instance_id": instance_id,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        r.setex(key, 30, json.dumps(data))  # 30s TTL
        return instance_id

    def deregister(self, service_name: str, instance_id: str) -> None:
        """Deregister a service instance."""
        r = self._get_redis()
        r.delete(f"lida:service:{service_name}:{instance_id}")

    def discover(self, service_name: str) -> List[Dict[str, Any]]:
        """Discover all instances of a service."""
        r = self._get_redis()
        pattern = f"lida:service:{service_name}:*"
        instances = []

        for key in r.scan_iter(pattern):
            data = r.get(key)
            if data:
                instances.append(json.loads(data))

        return instances

    def start_heartbeat(self, service_name: str, host: str, port: int, interval: float = 10.0) -> None:
        """Start background heartbeat thread."""
        self._running = True

        def heartbeat_loop():
            while self._running:
                try:
                    self.register(service_name, host, port)
                except Exception:
                    pass
                time.sleep(interval)

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        """Stop heartbeat thread."""
        self._running = False


# =============================================================================
# Pipeline Execution
# =============================================================================

class PipelineStep:
    """A step in a pipeline."""

    def __init__(
        self,
        name: str,
        command: List[str],
        condition: Optional[Callable[[], bool]] = None,
        on_failure: str = "abort",  # abort, continue, retry
        max_retries: int = 3,
        timeout: Optional[float] = None,
        environment: Dict[str, str] = None,
    ):
        self.name = name
        self.command = command
        self.condition = condition
        self.on_failure = on_failure
        self.max_retries = max_retries
        self.timeout = timeout
        self.environment = environment or {}


@dataclass
class StepResult:
    """Result of a pipeline step."""
    name: str
    success: bool
    return_code: int
    stdout: str
    stderr: str
    duration: float
    attempts: int


class Pipeline:
    """Execute a series of steps with error handling."""

    def __init__(self, name: str, working_dir: Optional[Path] = None):
        self.name = name
        self.working_dir = working_dir or Path.cwd()
        self.steps: List[PipelineStep] = []
        self.results: List[StepResult] = []

    def add_step(self, step: PipelineStep) -> "Pipeline":
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def add(
        self,
        name: str,
        command: List[str],
        **kwargs
    ) -> "Pipeline":
        """Convenience method to add a step."""
        self.steps.append(PipelineStep(name=name, command=command, **kwargs))
        return self

    def run(self, dry_run: bool = False) -> Tuple[bool, List[StepResult]]:
        """Execute the pipeline."""
        self.results = []

        for step in self.steps:
            # Check condition
            if step.condition and not step.condition():
                continue

            if dry_run:
                print(f"[DRY RUN] {step.name}: {' '.join(step.command)}")
                continue

            # Execute with retries
            attempts = 0
            result = None

            while attempts < step.max_retries:
                attempts += 1
                start = time.time()

                try:
                    env = os.environ.copy()
                    env.update(step.environment)

                    proc = subprocess.run(
                        step.command,
                        cwd=str(self.working_dir),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=step.timeout,
                    )

                    result = StepResult(
                        name=step.name,
                        success=proc.returncode == 0,
                        return_code=proc.returncode,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        duration=time.time() - start,
                        attempts=attempts,
                    )

                    if result.success:
                        break

                    if step.on_failure == "retry" and attempts < step.max_retries:
                        time.sleep(1)
                        continue
                    break

                except subprocess.TimeoutExpired:
                    result = StepResult(
                        name=step.name,
                        success=False,
                        return_code=-1,
                        stdout="",
                        stderr="Timeout",
                        duration=step.timeout or 0,
                        attempts=attempts,
                    )
                    break
                except Exception as e:
                    result = StepResult(
                        name=step.name,
                        success=False,
                        return_code=-1,
                        stdout="",
                        stderr=str(e),
                        duration=time.time() - start,
                        attempts=attempts,
                    )
                    break

            self.results.append(result)

            if not result.success and step.on_failure == "abort":
                return False, self.results

        all_success = all(r.success for r in self.results)
        return all_success, self.results

    def summary(self) -> str:
        """Generate execution summary."""
        lines = [f"Pipeline: {self.name}", "=" * 50]

        total_duration = sum(r.duration for r in self.results)
        success_count = sum(1 for r in self.results if r.success)

        for r in self.results:
            status = "✓" if r.success else "✗"
            lines.append(f"  {status} {r.name} ({r.duration:.2f}s, {r.attempts} attempt(s))")
            if not r.success and r.stderr:
                lines.append(f"      Error: {r.stderr[:100]}")

        lines.append("-" * 50)
        lines.append(f"Total: {success_count}/{len(self.results)} passed in {total_duration:.2f}s")

        return "\n".join(lines)


# =============================================================================
# Metrics & Monitoring
# =============================================================================

@dataclass
class Metric:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


class MetricsCollector:
    """Collect and expose metrics."""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self._metrics: Dict[str, List[Metric]] = {}
        self._lock = threading.Lock()
        self._max_history = 1000

    def record(self, name: str, value: float, labels: Dict[str, str] = None, unit: str = "") -> None:
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            unit=unit,
        )

        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(metric)

            # Trim history
            if len(self._metrics[name]) > self._max_history:
                self._metrics[name] = self._metrics[name][-self._max_history:]

    def get(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """Get metric history."""
        with self._lock:
            metrics = self._metrics.get(name, [])
            if since:
                metrics = [m for m in metrics if m.timestamp > since]
            return list(metrics)

    def get_latest(self, name: str) -> Optional[Metric]:
        """Get most recent value."""
        with self._lock:
            metrics = self._metrics.get(name, [])
            return metrics[-1] if metrics else None

    def get_stats(self, name: str, window: Optional[timedelta] = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        metrics = self.get(name)
        if window:
            cutoff = datetime.now() - window
            metrics = [m for m in metrics if m.timestamp > cutoff]

        if not metrics:
            return {}

        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1],
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        with self._lock:
            for name, metrics in self._metrics.items():
                if not metrics:
                    continue

                latest = metrics[-1]
                metric_name = name.replace(".", "_").replace("-", "_")

                labels_str = ""
                if latest.labels:
                    pairs = [f'{k}="{v}"' for k, v in latest.labels.items()]
                    labels_str = "{" + ",".join(pairs) + "}"

                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name}{labels_str} {latest.value}")

        return "\n".join(lines)

    def push_to_redis(self) -> None:
        """Push metrics to Redis for aggregation."""
        if not self.redis_url:
            return

        try:
            import redis
            r = redis.from_url(self.redis_url)

            hostname = socket.gethostname()
            timestamp = datetime.now().isoformat()

            with self._lock:
                for name, metrics in self._metrics.items():
                    if metrics:
                        latest = metrics[-1]
                        key = f"lida:metrics:{hostname}:{name}"
                        r.setex(key, 60, json.dumps({
                            "value": latest.value,
                            "timestamp": timestamp,
                            "labels": latest.labels,
                        }))
        except Exception:
            pass


# =============================================================================
# Cluster Management
# =============================================================================

@dataclass
class ClusterNode:
    """A node in the cluster."""
    hostname: str
    ssh_host: str
    redis_port: int = 6379
    api_port: int = 2040
    user: Optional[str] = None
    status: str = "unknown"
    last_seen: Optional[datetime] = None


class ClusterManager:
    """Manage a cluster of LIDA instances."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or Path.home() / ".config" / "lida" / "cluster.json"
        self.nodes: Dict[str, ClusterNode] = {}
        self._load()

    def _load(self) -> None:
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    data = json.load(f)
                for name, ndata in data.get("nodes", {}).items():
                    self.nodes[name] = ClusterNode(**ndata)
            except Exception:
                pass

    def _save(self) -> None:
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"nodes": {}}
        for name, node in self.nodes.items():
            data["nodes"][name] = {
                "hostname": node.hostname,
                "ssh_host": node.ssh_host,
                "redis_port": node.redis_port,
                "api_port": node.api_port,
                "user": node.user,
            }
        with open(self.config_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_node(self, name: str, ssh_host: str, **kwargs) -> ClusterNode:
        """Add a node to the cluster."""
        node = ClusterNode(
            hostname=name,
            ssh_host=ssh_host,
            **kwargs
        )
        self.nodes[name] = node
        self._save()
        return node

    def remove_node(self, name: str) -> bool:
        """Remove a node from the cluster."""
        if name in self.nodes:
            del self.nodes[name]
            self._save()
            return True
        return False

    def _ssh_cmd(self, node: ClusterNode, command: str) -> Tuple[int, str, str]:
        """Execute command on remote node via SSH."""
        ssh_target = f"{node.user}@{node.ssh_host}" if node.user else node.ssh_host

        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", ssh_target, command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, result.stdout, result.stderr

    def check_status(self, name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Check status of cluster nodes."""
        nodes_to_check = [self.nodes[name]] if name else list(self.nodes.values())
        results = {}

        for node in nodes_to_check:
            try:
                # Check SSH connectivity
                code, stdout, stderr = self._ssh_cmd(node, "echo OK")
                if code != 0:
                    results[node.hostname] = {"status": "unreachable", "error": stderr}
                    continue

                # Check LIDA services
                code, stdout, _ = self._ssh_cmd(
                    node,
                    f"curl -s http://localhost:{node.api_port}/health 2>/dev/null || echo FAIL"
                )

                api_healthy = "FAIL" not in stdout

                results[node.hostname] = {
                    "status": "healthy" if api_healthy else "degraded",
                    "api_port": node.api_port,
                    "redis_port": node.redis_port,
                    "api_healthy": api_healthy,
                }
                node.status = results[node.hostname]["status"]
                node.last_seen = datetime.now()

            except Exception as e:
                results[node.hostname] = {"status": "error", "error": str(e)}

        return results

    def deploy(self, name: str, command: str = "lida system start") -> Tuple[bool, str]:
        """Deploy/start LIDA on a remote node."""
        node = self.nodes.get(name)
        if not node:
            return False, f"Unknown node: {name}"

        full_cmd = f"cd ~/lida-multiagents-research && {command} --redis-port {node.redis_port} --api-port {node.api_port}"
        code, stdout, stderr = self._ssh_cmd(node, full_cmd)

        return code == 0, stdout if code == 0 else stderr

    def stop_all(self) -> Dict[str, bool]:
        """Stop LIDA on all nodes."""
        results = {}
        for name, node in self.nodes.items():
            success, _ = self.deploy(name, "lida system stop")
            results[name] = success
        return results


# =============================================================================
# File-based Lock (for local coordination)
# =============================================================================

class FileLock:
    """Simple file-based lock for local process coordination."""

    def __init__(self, lock_file: Path):
        self.lock_file = lock_file
        self._fd = None

    @contextmanager
    def acquire(self, timeout: float = 10.0, blocking: bool = True):
        """Acquire the file lock."""
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        self._fd = open(self.lock_file, "w")

        start = time.time()
        acquired = False

        while True:
            try:
                flags = fcntl.LOCK_EX
                if not blocking:
                    flags |= fcntl.LOCK_NB
                fcntl.flock(self._fd, flags)
                acquired = True
                break
            except BlockingIOError:
                if not blocking or (time.time() - start > timeout):
                    break
                time.sleep(0.1)

        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            self._fd.close()
            self._fd = None


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_pipeline(project_root: Path) -> Pipeline:
    """Create a default deployment pipeline."""
    pipeline = Pipeline("deploy", project_root)

    pipeline.add("lint", ["ruff", "check", "src/"], on_failure="continue")
    pipeline.add("test", ["pytest", "tests/", "-v", "--tb=short"], on_failure="abort")
    pipeline.add("start-redis", ["redis-server", "--daemonize", "yes"])
    pipeline.add("start-api", [sys.executable, "run_swarm_server.py", "--port=2040"])

    return pipeline


def quick_start(redis_port: int = 6379, api_port: int = 2040) -> ServiceOrchestrator:
    """Quick start all services."""
    project_root = Path(__file__).parent.parent.parent
    orch = ServiceOrchestrator(project_root)
    orch.register_default_services(redis_port, api_port)
    orch.start()
    return orch
