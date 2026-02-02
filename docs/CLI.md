# LIDA CLI Reference

## Overview

LIDA (Large-scale Intelligent Deliberation Architecture) provides a comprehensive command-line interface for running multi-agent simulations, managing distributed systems, and conducting AI safety research experiments.

```
lida <command> [subcommand] [options]
```

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Commands](#core-commands)
3. [System Management](#system-management)
4. [Advanced Orchestration](#advanced-orchestration)
5. [Observability & Monitoring](#observability--monitoring)
6. [Fault Tolerance](#fault-tolerance)
7. [Configuration Management](#configuration-management)
8. [Cluster Operations](#cluster-operations)
9. [Development & CI/CD](#development--cicd)
10. [Architecture](#architecture)

---

## Quick Start

### Installation

```bash
# Install with all dependencies
pip install -e ".[all]"

# Verify installation
lida --version
```

### First Run

```bash
# Start the system
lida system start

# Run a quick demo
lida demo --type quick

# Check system health
lida health check

# Stop the system
lida system stop
```

### Multi-User Setup (Mila Cluster)

```bash
# User 1 (default ports)
eval $(lida system env --redis-port 6379 --api-port 2040)
lida system start

# User 2 (offset ports)
eval $(lida system env --redis-port 6380 --api-port 2041)
lida system start --redis-port 6380 --api-port 2041

# User 3
eval $(lida system env --redis-port 6381 --api-port 2042)
lida system start --redis-port 6381 --api-port 2042
```

---

## Core Commands

### `lida run`

Run a debate scenario with configurable parameters.

```bash
lida run <scenario> [options]
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `scenario` | Scenario name or path to YAML file |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--topic` | from scenario | Override debate topic |
| `--rounds` | 5 | Maximum number of rounds |
| `--timeout` | 300 | Timeout in seconds |
| `--no-live` | false | Disable LLM calls (simulation mode) |
| `--output` | auto | Output directory for results |

**Examples:**
```bash
# Run AI x-risk debate
lida run ai_xrisk

# Run with custom topic
lida run ai_xrisk --topic "Should we pause AI development?"

# Simulation mode (no API calls)
lida run ai_xrisk --no-live --rounds 10
```

---

### `lida simulate`

Run policy simulations with game-theoretic dynamics.

```bash
lida simulate <scenario> [options]
```

**Scenarios:**
| Scenario | Description |
|----------|-------------|
| `chip_war` | US-China semiconductor competition |
| `agi_crisis` | AGI development race dynamics |
| `negotiation` | Two-party negotiation simulation |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--ticks` | 10 | Number of simulation ticks |
| `--agent-a` | - | First agent for negotiation |
| `--agent-b` | - | Second agent for negotiation |
| `-v, --verbose` | false | Verbose output |

**Examples:**
```bash
# Run chip war simulation
lida simulate chip_war --ticks 20

# Run negotiation between specific agents
lida simulate negotiation --agent-a sam_altman --agent-b elon_musk
```

---

### `lida quorum`

Run multi-agent quorum deliberation with various presets.

```bash
lida quorum [options]
```

**Presets:**
| Preset | Description |
|--------|-------------|
| `realtime` | Real-time quorum with simulated industrial events |
| `gdelt` | Live GDELT news feed (updates every 15 min) |
| `mlx` | MLX streaming backend (Apple Silicon optimized) |
| `openrouter` | OpenRouter API backend |
| `advanced` | Full configuration mode |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--preset, -p` | realtime | Quorum preset |
| `--event, -e` | - | Event headline to analyze |
| `--backend` | openrouter | LLM backend (openrouter, mlx) |
| `--duration` | 60 | Duration in seconds |
| `--cycles` | 5 | Number of update cycles |
| `--watch` | - | Companies to watch (comma-separated) |
| `--test` | false | Quick test mode |

**Examples:**
```bash
# Real-time quorum for 2 minutes
lida quorum --preset realtime --duration 120

# GDELT news monitoring
lida quorum --preset gdelt --cycles 10 --watch nvidia,openai,anthropic

# Analyze specific event
lida quorum --preset mlx --event "Major AI lab announces AGI breakthrough"
```

---

### `lida debate`

Run structured AI safety debates with predefined topics and matchups.

```bash
lida debate [options]
```

**Topics:**
| Topic ID | Description |
|----------|-------------|
| `ai_pause` | Should AI development be paused? |
| `lab_self_regulation` | Can AI labs self-regulate effectively? |
| `xrisk_vs_present_harms` | Existential risk vs. present-day harms |
| `scaling_hypothesis` | Will scaling lead to AGI? |
| `open_source_ai` | Should AI models be open source? |
| `government_regulation` | Role of government in AI regulation |

**Matchups:**
| Matchup ID | Participants |
|------------|--------------|
| `doom_vs_accel` | Doomers vs Accelerationists |
| `labs_debate` | AI Lab representatives |
| `academics_clash` | Academic researchers |
| `ethics_vs_scale` | Ethicists vs Scaling advocates |
| `full_panel` | All perspectives |

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--topic, -t` | - | Debate topic ID |
| `--matchup, -m` | - | Predefined matchup |
| `--rounds, -r` | 5 | Number of debate rounds |
| `--interactive, -i` | false | Interactive mode |
| `--auto, -a` | false | Auto-run without prompts |
| `--no-llm` | false | Run without LLM calls |
| `--provider` | - | LLM provider |
| `--model` | - | Specific model ID |
| `--list` | false | List available options |

**Examples:**
```bash
# List all topics and matchups
lida debate --list

# Run doomers vs accelerationists debate
lida debate --matchup doom_vs_accel --auto

# Custom topic with specific model
lida debate --topic ai_pause --rounds 6 --model anthropic/claude-sonnet-4

# Interactive debate
lida debate --interactive
```

---

### `lida demo`

Run demonstration scenarios.

```bash
lida demo [options]
```

**Demo Types:**
| Type | Description |
|------|-------------|
| `quick` | Quick streaming demo (default) |
| `live` | Live demo with GDELT + streaming |
| `streaming` | Streaming demo with real-time output |
| `swarm` | Live swarm behavior demo |
| `persuasion` | Persuasion experiment demo |
| `hyperdash` | Hyperdimensional dashboard demo |

**Examples:**
```bash
lida demo                    # Quick demo
lida demo --type live        # Live GDELT demo
lida demo --type swarm       # Swarm behavior
```

---

### `lida chat`

Two-persona conversation simulation.

```bash
lida chat <persona1> <persona2> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--topic, -t` | auto | Conversation topic |
| `--turns` | 5 | Number of conversation turns |

**Examples:**
```bash
lida chat sam_altman elon_musk --topic "AI safety" --turns 10
lida chat eliezer_yudkowsky marc_andreessen
```

---

## System Management

### `lida system`

Manage LIDA system services with port configuration for multi-user environments.

#### `lida system start`

Start LIDA services (Redis, API server, workers).

```bash
lida system start [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--redis-port` | 6379 | Redis server port |
| `--api-port` | 2040 | API server port |
| `--workers, -w` | - | Number of worker processes |
| `--worker-replicas` | 1 | Worker container replicas (Docker) |
| `--agents, -a` | - | Number of agents |
| `--scenario, -s` | - | Scenario to load |
| `--live` | false | Enable live LLM mode |
| `--docker` | false | Use Docker (default: native) |
| `--force, -f` | false | Start even if ports in use |

**Examples:**
```bash
# Start with defaults
lida system start

# Start on custom ports
lida system start --redis-port 6380 --api-port 2041

# Start with workers and live mode
lida system start --workers 8 --live

# Force start with Docker
lida system start --docker --force
```

#### `lida system stop`

Stop LIDA services.

```bash
lida system stop [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--redis-port` | 6379 | Redis port to stop |
| `--api-port` | 2040 | API port to stop |
| `--docker` | false | Stop Docker containers |
| `--include-redis` | false | Also stop Redis |

#### `lida system status`

Show system status and port usage.

```bash
lida system status
```

**Output:**
```
============================================================
 LIDA System Status
============================================================

Port Status:
  ● 6379   Redis (default)       [IN USE]
  ○ 6380   Redis (user 2)        [available]
  ○ 6381   Redis (user 3)        [available]
  ● 2040   API (default)         [IN USE]
  ○ 2041   API (user 2)          [available]
  ○ 2042   API (user 3)          [available]

--------------------------------------------------
 Running Processes
--------------------------------------------------
  redis-server --port 6379...
  python run_swarm_server.py --port=2040...

--------------------------------------------------
 Multi-User Port Assignments
--------------------------------------------------
  User 1: --redis-port 6379 --api-port 2040
  User 2: --redis-port 6380 --api-port 2041
  User 3: --redis-port 6381 --api-port 2042
```

#### `lida system env`

Print environment variables for shell configuration.

```bash
lida system env [options]
```

**Usage:**
```bash
# Print environment
lida system env --redis-port 6380 --api-port 2041

# Source directly
eval $(lida system env --redis-port 6380 --api-port 2041)
```

**Output:**
```bash
# LIDA environment for Redis:6380 API:2041
export REDIS_PORT=6380
export REDIS_URL=redis://localhost:6380
export API_PORT=2041
export PORT=2041

# Run: eval $(lida system env --redis-port ... --api-port ...)
```

---

### `lida serve`

Start the LIDA API server directly.

```bash
lida serve [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 2040 | Port number |
| `--workers` | 1 | Uvicorn worker processes |
| `--reload` | false | Auto-reload on changes |
| `--scenario, -s` | - | Scenario to load |
| `--live` | false | Enable live LLM mode |
| `--redis-url` | - | Redis URL for messaging |
| `--agents, -a` | - | Number of agents |
| `--advanced` | false | Use advanced swarm server |
| `--simple` | false | Use simple server (no swarm) |

**Examples:**
```bash
# Basic server
lida serve --port 8080

# Production mode with multiple workers
lida serve --port 8080 --workers 4 --live

# Development mode with auto-reload
lida serve --port 8080 --reload
```

---

### `lida workers`

Run background worker pool for distributed task processing.

```bash
lida workers [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--count, -n` | 4 | Number of workers |
| `--redis-url` | redis://localhost:6379 | Redis URL |
| `--capacity` | 5 | Tasks per worker |
| `--work-types` | general,compute,io,analysis,llm | Worker specializations |

**Examples:**
```bash
# Start 8 workers
lida workers --count 8

# Workers for specific task types
lida workers --count 4 --work-types llm,analysis

# Connect to different Redis
lida workers --redis-url redis://localhost:6380
```

---

### `lida deliberate`

Run a deliberation against a running LIDA instance.

```bash
lida deliberate [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--port, -p` | required | API port of running instance |
| `--topic, -t` | - | Deliberation topic |
| `--scenario, -s` | quick_personas3 | Scenario name |
| `--timeout` | 0 | Timeout in seconds (0=infinite) |
| `--poll-interval` | 5 | Status poll interval |

**Examples:**
```bash
# Run deliberation on default port
lida deliberate --port 2040 --topic "Should AI development be regulated?"

# Run with timeout
lida deliberate --port 2041 --topic "AI safety" --timeout 300

# Use specific scenario
lida deliberate --port 2040 --scenario full_panel
```

---

## Advanced Orchestration

### `lida orchestrate`

Advanced service orchestration with automatic dependency management.

#### Features:
- **Dependency Resolution**: Services start in correct order (Redis before API)
- **Health Monitoring**: Continuous health checks with auto-restart
- **Graceful Shutdown**: Services stop in reverse dependency order

#### `lida orchestrate start`

```bash
lida orchestrate start [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--redis-port` | 6379 | Redis port |
| `--api-port` | 2040 | API port |
| `--workers` | - | Number of workers |
| `--monitor, -m` | false | Enable health monitoring |

**Examples:**
```bash
# Start with orchestration
lida orchestrate start --redis-port 6379 --api-port 2040

# Start with continuous monitoring
lida orchestrate start --monitor

# Start with workers
lida orchestrate start --workers 4 --monitor
```

#### `lida orchestrate status`

```bash
lida orchestrate status
```

**Output:**
```
============================================================
 Service Status
============================================================

  ● redis        RUNNING     PID:12345
  ● api          RUNNING     PID:12346
  ○ workers      STOPPED     PID:-
```

#### `lida orchestrate stop`

```bash
lida orchestrate stop
```

---

### `lida profile`

Manage named configuration profiles for different environments.

#### `lida profile create`

```bash
lida profile create <name> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--redis-port` | 6379 | Redis port |
| `--api-port` | 2040 | API port |
| `--workers` | - | Number of workers |
| `--live` | false | Enable live mode |
| `--scenario, -s` | - | Default scenario |
| `--description, -d` | - | Profile description |

**Examples:**
```bash
# Create development profile
lida profile create dev --redis-port 6379 --api-port 2040 --description "Local dev"

# Create production profile
lida profile create prod --redis-port 6380 --api-port 2041 --workers 8 --live
```

#### `lida profile list`

```bash
lida profile list
```

**Output:**
```
============================================================
 Configuration Profiles
============================================================

  → dev              Redis:6379  API:2040  Workers:4
      Local development environment
    prod             Redis:6380  API:2041  Workers:8
      Production configuration
```

#### `lida profile use`

Set the default profile.

```bash
lida profile use <name>
```

#### `lida profile env`

Export profile as shell environment variables.

```bash
eval $(lida profile env dev)
```

---

## Observability & Monitoring

### `lida health`

Comprehensive health checking for all system components.

#### `lida health check`

```bash
lida health check [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--name, -n` | all | Specific check to run |
| `--redis-port` | 6379 | Redis port to check |
| `--api-port` | 2040 | API port to check |

**Checks Performed:**
| Check | Description |
|-------|-------------|
| `redis` | Redis connectivity |
| `api` | API server health endpoint |
| `disk` | Disk space usage |

**Output:**
```
============================================================
 Health Check Results
============================================================

  ● redis           [HEALTHY   ] Redis responding on port 6379
  ● api             [HEALTHY   ] API healthy on port 2040
  ◐ disk            [DEGRADED  ] Disk usage high: 85.2%
```

**Exit Codes:**
| Code | Meaning |
|------|---------|
| 0 | All checks healthy |
| 1 | One or more checks unhealthy |

#### `lida health watch`

Continuous health monitoring.

```bash
lida health watch --interval 10
```

#### `lida health json`

Output health status as JSON (for scripts/monitoring).

```bash
lida health json
```

**Output:**
```json
{
  "status": "healthy",
  "checks": {
    "redis": {"status": "HEALTHY", "message": "Redis responding on port 6379"},
    "api": {"status": "HEALTHY", "message": "API healthy on port 2040"},
    "disk": {"status": "HEALTHY", "message": "Disk usage: 45.2%"}
  }
}
```

---

### `lida metrics`

View and export system metrics.

#### `lida metrics show`

```bash
lida metrics show
```

#### `lida metrics watch`

Real-time metrics monitoring.

```bash
lida metrics watch --interval 5
```

#### `lida metrics export`

Export metrics in Prometheus format.

```bash
lida metrics export --output metrics.prom
```

**Output Format:**
```
# TYPE service_redis_up gauge
service_redis_up{port="6379"} 1.0
# TYPE service_api_up gauge
service_api_up{port="2040"} 1.0
```

---

### `lida trace`

Distributed tracing for debugging and performance analysis.

#### `lida trace start`

Start a new trace.

```bash
lida trace start --operation "deliberation-run"
```

**Output:**
```
Started trace: abc123def456
Span ID: span_001
Operation: deliberation-run
```

#### `lida trace finish`

Finish the current trace.

```bash
lida trace finish
```

#### `lida trace show`

Display trace details.

```bash
lida trace show --trace-id abc123def456
```

**Output:**
```
============================================================
 Trace: abc123def456
============================================================

Service: lida-cli
Spans: 5

├─ deliberation-run [OK] 1523.45ms
  ├─ load-personas [OK] 45.23ms
  ├─ initialize-agents [OK] 123.45ms
  ├─ run-rounds [OK] 1234.56ms
  └─ save-results [OK] 120.21ms
```

#### `lida trace export`

Export trace as JSON.

```bash
lida trace export --trace-id abc123def456 --output trace.json
```

---

### `lida events`

Event bus monitoring and publishing for inter-service communication.

#### `lida events history`

View recent events.

```bash
lida events history [--type <event_type>] [--verbose]
```

**Examples:**
```bash
# All recent events
lida events history

# Filter by type
lida events history --type chaos.enabled

# Verbose output
lida events history --verbose
```

#### `lida events watch`

Watch events in real-time.

```bash
lida events watch
```

**Output:**
```
Watching events (Ctrl+C to stop)...
[14:23:45] service.started: {"name": "api", "port": 2040}
[14:23:46] health.check: {"status": "healthy"}
[14:23:47] deliberation.started: {"topic": "AI safety"}
```

#### `lida events publish`

Publish a custom event.

```bash
lida events publish <event_type> --data '<json>'
```

**Examples:**
```bash
lida events publish my.custom.event --data '{"key": "value"}'
lida events publish alert.triggered --data "High CPU usage detected"
```

---

## Fault Tolerance

### `lida circuit`

Circuit breaker management for preventing cascading failures.

#### Circuit Breaker States

```
CLOSED ──(failures)──> OPEN ──(timeout)──> HALF_OPEN ──(success)──> CLOSED
   ▲                                            │
   └────────────────────(failure)───────────────┘
```

| State | Description |
|-------|-------------|
| `CLOSED` | Normal operation, requests pass through |
| `OPEN` | Service failing, requests rejected immediately |
| `HALF_OPEN` | Testing recovery, limited requests allowed |

#### `lida circuit status`

Show all circuit breakers.

```bash
lida circuit status
```

**Output:**
```
============================================================
 Circuit Breakers
============================================================

  ● api-service      [CLOSED    ] failures=0
  ○ redis-client     [OPEN      ] failures=5
  ◐ external-api     [HALF_OPEN ] failures=3
```

#### `lida circuit configure`

Configure a circuit breaker.

```bash
lida circuit configure <name> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--failure-threshold` | 5 | Failures before opening |
| `--recovery-timeout` | 30.0 | Seconds before half-open |

**Example:**
```bash
lida circuit configure api-service --failure-threshold 3 --recovery-timeout 60
```

#### `lida circuit reset`

Reset a circuit breaker to CLOSED state.

```bash
lida circuit reset <name>
```

#### `lida circuit trip`

Manually trip a circuit breaker (for testing).

```bash
lida circuit trip <name>
```

---

### `lida chaos`

Chaos engineering for resilience testing through controlled fault injection.

#### Fault Types

| Type | Description |
|------|-------------|
| `LATENCY` | Add artificial delay to operations |
| `ERROR` | Inject exceptions |
| `TIMEOUT` | Simulate operation timeouts |
| `RESOURCE_EXHAUSTION` | Simulate memory pressure |

#### `lida chaos enable`

Enable the chaos engine.

```bash
lida chaos enable
```

#### `lida chaos disable`

Disable the chaos engine.

```bash
lida chaos disable
```

#### `lida chaos add`

Add a fault configuration.

```bash
lida chaos add <name> --type <fault_type> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--type, -t` | required | Fault type |
| `--probability, -p` | 0.1 | Injection probability (0-1) |
| `--duration, -d` | - | Duration for latency/timeout |
| `--services, -s` | all | Target services (comma-separated) |

**Examples:**
```bash
# Add 10% latency fault
lida chaos add slow-api --type latency --probability 0.1 --duration 2.0

# Add 5% error fault for specific services
lida chaos add api-errors --type error --probability 0.05 --services api,workers

# Add timeout simulation
lida chaos add timeout-test --type timeout --probability 0.02 --duration 30
```

#### `lida chaos status`

Show chaos engine status.

```bash
lida chaos status
```

**Output:**
```
============================================================
 Chaos Engine Status
============================================================

  Active: YES

  Registered Faults:
    ● slow-api: LATENCY @ 10%
        Injections: 42
    ● api-errors: ERROR @ 5%
        Injections: 7
```

#### `lida chaos inject`

Manually inject a fault (for testing).

```bash
lida chaos inject <fault_name>
```

---

### `lida lock`

Distributed locking for cluster coordination using Redis.

#### `lida lock acquire`

Acquire a distributed lock.

```bash
lida lock acquire <name> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--timeout` | 30.0 | Lock acquisition timeout |
| `--command, -c` | - | Command to run while holding lock |

**Examples:**
```bash
# Acquire and hold until Enter
lida lock acquire deploy-lock

# Acquire, run command, release
lida lock acquire deploy-lock --command "lida system restart"

# With custom timeout
lida lock acquire critical-section --timeout 60
```

#### `lida lock status`

Check lock status.

```bash
lida lock status <name>
```

**Output:**
```
  ● deploy-lock: LOCKED
```

#### `lida lock release`

Force release a lock (admin operation).

```bash
lida lock release <name>
```

---

## Configuration Management

### `lida snapshot`

System state snapshot and restore for debugging or recovery.

#### `lida snapshot create`

Create a snapshot of current system state.

```bash
lida snapshot create <name> [--description <text>]
```

**Example:**
```bash
lida snapshot create pre-upgrade --description "State before v2.0 upgrade"
```

#### `lida snapshot list`

List all snapshots.

```bash
lida snapshot list
```

**Output:**
```
============================================================
 Snapshots
============================================================

  abc123  pre-upgrade           2024-01-15 14:30
  def456  after-config-change   2024-01-14 09:15
  ghi789  initial-setup         2024-01-10 11:00
```

#### `lida snapshot show`

Show snapshot details.

```bash
lida snapshot show <snapshot_id>
```

#### `lida snapshot restore`

Restore system state from a snapshot.

```bash
lida snapshot restore <snapshot_id>
```

#### `lida snapshot delete`

Delete a snapshot.

```bash
lida snapshot delete <snapshot_id>
```

---

### `lida schedule`

Job scheduler for recurring tasks.

#### `lida schedule add`

Add a scheduled job.

```bash
lida schedule add <name> --interval <seconds> --command "<command>"
```

**Examples:**
```bash
# Health check every minute
lida schedule add health-check --interval 60 --command "lida health check"

# Cleanup every hour
lida schedule add cleanup --interval 3600 --command "lida data cleanup"

# Metrics export every 5 minutes
lida schedule add metrics-export --interval 300 --command "lida metrics export -o /tmp/metrics.prom"
```

#### `lida schedule list`

List scheduled jobs.

```bash
lida schedule list
```

**Output:**
```
============================================================
 Scheduled Jobs
============================================================

  ● health-check         [every 60s]
      Next: 2024-01-15T14:31:00
  ● cleanup              [every 3600s]
      Next: 2024-01-15T15:00:00
```

#### `lida schedule start`

Start the scheduler daemon.

```bash
lida schedule start
```

#### `lida schedule remove`

Remove a scheduled job.

```bash
lida schedule remove <job_id>
```

---

## Cluster Operations

### `lida cluster`

Manage a cluster of LIDA instances across multiple machines.

#### `lida cluster add`

Add a node to the cluster.

```bash
lida cluster add <name> --host <ssh_host> [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--host` | required | SSH hostname |
| `--user, -u` | current | SSH username |
| `--redis-port` | 6379 | Redis port on node |
| `--api-port` | 2040 | API port on node |

**Examples:**
```bash
# Add node with defaults
lida cluster add node1 --host server1.mila.quebec

# Add node with custom user and ports
lida cluster add node2 --host server2.mila.quebec --user arthur --redis-port 6380 --api-port 2041
```

#### `lida cluster list`

List all cluster nodes.

```bash
lida cluster list
```

**Output:**
```
============================================================
 Cluster Nodes
============================================================

  ● node1           server1.mila.quebec       API:2040
  ◐ node2           server2.mila.quebec       API:2041
  ○ node3           server3.mila.quebec       API:2042
```

#### `lida cluster status`

Check cluster health.

```bash
lida cluster status [node_name]
```

#### `lida cluster deploy`

Deploy LIDA to a remote node.

```bash
lida cluster deploy <name> [--command "<custom_command>"]
```

**Examples:**
```bash
# Deploy with defaults
lida cluster deploy node1

# Custom deployment command
lida cluster deploy node2 --command "lida system start --live"
```

#### `lida cluster stop-all`

Stop LIDA on all cluster nodes.

```bash
lida cluster stop-all
```

#### `lida cluster remove`

Remove a node from the cluster.

```bash
lida cluster remove <name>
```

---

## Development & CI/CD

### `lida pipeline`

Run deployment and CI/CD pipelines.

#### Presets

| Preset | Steps |
|--------|-------|
| `deploy` | lint → test → start services |
| `test` | lint → typecheck → test |
| `ci` | install → lint → test |

#### `lida pipeline --preset`

Run a preset pipeline.

```bash
lida pipeline --preset <preset> [--dry-run]
```

**Examples:**
```bash
# Run test pipeline
lida pipeline --preset test

# Dry run deployment
lida pipeline --preset deploy --dry-run

# CI pipeline
lida pipeline --preset ci
```

#### `lida pipeline --steps`

Run custom pipeline steps.

```bash
lida pipeline --steps "<cmd1>" "<cmd2>" ...
```

**Example:**
```bash
lida pipeline --steps "pytest tests/ -v" "lida system start" "lida health check"
```

**Output:**
```
============================================================
 Pipeline: default
============================================================

[DRY RUN] lint: ruff check src/
[DRY RUN] typecheck: mypy src/
[DRY RUN] test: pytest tests/ -v

Pipeline: default
==================================================
  ✓ lint (0.45s, 1 attempt(s))
  ✓ typecheck (2.31s, 1 attempt(s))
  ✓ test (12.45s, 1 attempt(s))
--------------------------------------------------
Total: 3/3 passed in 15.21s
```

---

### `lida autoscale`

Auto-scaling configuration based on system metrics.

#### `lida autoscale configure`

Configure auto-scaling policy.

```bash
lida autoscale configure [options]
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--min` | 1 | Minimum instances |
| `--max` | 10 | Maximum instances |
| `--scale-up-threshold` | 80.0 | CPU% to trigger scale up |
| `--scale-down-threshold` | 30.0 | CPU% to trigger scale down |

**Example:**
```bash
lida autoscale configure --min 2 --max 8 --scale-up-threshold 70
```

#### `lida autoscale status`

Show auto-scaler status.

```bash
lida autoscale status
```

#### `lida autoscale simulate`

Simulate auto-scaling decisions.

```bash
lida autoscale simulate --min 1 --max 5
```

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                        LIDA CLI                              │
├─────────────────────────────────────────────────────────────┤
│  Core Commands    │  System Mgmt    │  Advanced Features    │
│  - run            │  - system       │  - orchestrate        │
│  - simulate       │  - serve        │  - chaos              │
│  - quorum         │  - workers      │  - circuit            │
│  - debate         │  - deliberate   │  - trace              │
│  - demo           │                 │  - autoscale          │
│  - chat           │                 │  - health             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
├───────────────┬───────────────┬─────────────────────────────┤
│    Redis      │   API Server  │        Workers              │
│   (Pub/Sub)   │   (FastAPI)   │    (Task Processing)        │
└───────────────┴───────────────┴─────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent Framework                           │
├───────────────┬───────────────┬─────────────────────────────┤
│   Personas    │  Deliberation │      LLM Backends           │
│   Manager     │    Engine     │  (OpenRouter, MLX, etc.)    │
└───────────────┴───────────────┴─────────────────────────────┘
```

### Port Allocation

| Service | Default | User 2 | User 3 |
|---------|---------|--------|--------|
| Redis | 6379 | 6380 | 6381 |
| API Server | 2040 | 2041 | 2042 |
| Dashboard | 12345 | 12346 | 12347 |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `REDIS_PORT` | Redis server port |
| `REDIS_URL` | Full Redis URL |
| `API_PORT` | API server port |
| `PORT` | Alias for API_PORT |
| `SCENARIO` | Default scenario name |
| `SWARM_LIVE` | Enable live LLM mode |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |

---

## Appendix

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 130 | Interrupted (Ctrl+C) |

### Configuration Files

| File | Location | Purpose |
|------|----------|---------|
| Profiles | `~/.config/lida/profiles.json` | Named configurations |
| Cluster | `~/.config/lida/cluster.json` | Cluster node definitions |
| Snapshots | `~/.lida/snapshots/` | System state snapshots |
| Autoscale | `~/.lida/autoscale.json` | Auto-scaling policy |

### Troubleshooting

#### Port Already in Use

```bash
# Check what's using the port
lida system status

# Force start on different port
lida system start --api-port 2041 --force
```

#### Redis Connection Failed

```bash
# Check Redis health
lida health check --name redis

# Start Redis manually
redis-server --port 6379
```

#### API Not Responding

```bash
# Check API health
lida health check --name api

# Check logs
lida system status

# Restart with verbose output
lida serve --port 2040 --reload
```

---

*Generated for LIDA v0.1.0*
