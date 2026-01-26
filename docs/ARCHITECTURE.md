# LIDA Architecture

## System Overview

LIDA (Large-scale Intelligent Deliberation Architecture) is a distributed multi-agent system designed for AI safety research, policy simulation, and structured deliberation.

```
                                    ┌─────────────────────────────────────┐
                                    │           External APIs             │
                                    │  (OpenRouter, Anthropic, OpenAI)    │
                                    └──────────────────┬──────────────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                                  LIDA CLI                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │    Core     │  │   System    │  │  Advanced   │  │    Observability    │  │
│  │  Commands   │  │ Management  │  │  Features   │  │    & Monitoring     │  │
│  │             │  │             │  │             │  │                     │  │
│  │ run         │  │ system      │  │ orchestrate │  │ health              │  │
│  │ simulate    │  │ serve       │  │ profile     │  │ metrics             │  │
│  │ quorum      │  │ workers     │  │ cluster     │  │ trace               │  │
│  │ debate      │  │ deliberate  │  │ pipeline    │  │ events              │  │
│  │ demo        │  │             │  │ chaos       │  │ circuit             │  │
│  │ chat        │  │             │  │ autoscale   │  │ snapshot            │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
          ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
          │   API Server    │ │     Redis       │ │    Workers      │
          │   (FastAPI)     │ │   (Pub/Sub)     │ │  (Task Queue)   │
          │                 │ │                 │ │                 │
          │ Port: 2040      │ │ Port: 6379      │ │ Async Tasks     │
          │ REST + WebSocket│ │ Message Bus     │ │ LLM Calls       │
          │ SSE Streaming   │ │ State Store     │ │ Analysis        │
          └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                   │                   │                   │
                   └───────────────────┼───────────────────┘
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────────┐
          │                     Agent Framework                         │
          │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
          │  │   Personas   │  │ Deliberation │  │   LLM Backend    │  │
          │  │   Manager    │  │   Engine     │  │   Abstraction    │  │
          │  │              │  │              │  │                  │  │
          │  │ 169 personas │  │ Quorum       │  │ OpenRouter       │  │
          │  │ Belief state │  │ Debate       │  │ Anthropic        │  │
          │  │ Memory       │  │ Consensus    │  │ OpenAI           │  │
          │  │ Forking      │  │ Voting       │  │ MLX (local)      │  │
          │  └──────────────┘  └──────────────┘  └──────────────────┘  │
          └────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
          ┌────────────────────────────────────────────────────────────┐
          │                    Analysis Layer                           │
          │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
          │  │   Causal     │  │ Counterfact- │  │    Mechanism     │  │
          │  │   Engine     │  │    ual       │  │    Discovery     │  │
          │  └──────────────┘  └──────────────┘  └──────────────────┘  │
          └────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. CLI Layer (`src/cli/`)

The CLI provides the primary interface for interacting with LIDA.

```
src/cli/
├── main.py           # Main CLI entry point, argument parsing
├── runner.py         # Experiment runner
├── config_loader.py  # YAML configuration loading
├── progress.py       # Terminal UI progress display
├── sweep.py          # Parameter sweep functionality
├── advanced.py       # Advanced features v1 (profiles, orchestration)
└── advanced_v2.py    # Advanced features v2 (chaos, tracing, circuit breakers)
```

#### Key Classes

| Class | File | Purpose |
|-------|------|---------|
| `CLIError` | main.py | CLI-specific exception with exit code |
| `ExperimentRunner` | runner.py | Orchestrates experiment execution |
| `ProfileManager` | advanced.py | Manages configuration profiles |
| `ServiceOrchestrator` | advanced.py | Service lifecycle management |
| `CircuitBreaker` | advanced_v2.py | Fault tolerance pattern |
| `ChaosEngine` | advanced_v2.py | Fault injection for testing |
| `Tracer` | advanced_v2.py | Distributed tracing |
| `AutoScaler` | advanced_v2.py | Dynamic scaling based on metrics |

---

### 2. API Server Layer

The API server provides REST and WebSocket interfaces.

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Server                               │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │    REST     │  │  WebSocket  │  │         SSE             │  │
│  │  Endpoints  │  │   Handler   │  │   (Server-Sent Events)  │  │
│  │             │  │             │  │                         │  │
│  │ /api/       │  │ /ws         │  │ /stream                 │  │
│  │ /health     │  │ Real-time   │  │ Streaming responses     │  │
│  │ /deliberate │  │ updates     │  │ Live agent output       │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Middleware Stack                        │  │
│  │  CORS │ Rate Limiting │ Authentication │ Request Logging  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/stats` | GET | System statistics |
| `/api/deliberate` | POST | Start deliberation |
| `/api/deliberation/status` | GET | Get deliberation status |
| `/api/personas` | GET | List personas |
| `/api/scenarios` | GET | List scenarios |
| `/ws` | WebSocket | Real-time updates |
| `/stream` | GET (SSE) | Streaming responses |

---

### 3. Message Bus (Redis)

Redis provides pub/sub messaging and state storage.

```
┌─────────────────────────────────────────────────────────────────┐
│                          Redis                                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Pub/Sub Channels                      │    │
│  │                                                          │    │
│  │  lida:events          - System events                    │    │
│  │  lida:deliberation:*  - Deliberation updates             │    │
│  │  lida:agent:*         - Agent messages                   │    │
│  │  lida:metrics         - Metrics updates                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    Key-Value Store                       │    │
│  │                                                          │    │
│  │  lida:state:*         - System state                     │    │
│  │  lida:lock:*          - Distributed locks                │    │
│  │  lida:service:*       - Service registry                 │    │
│  │  lida:metrics:*       - Metrics time series              │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      Task Queue                          │    │
│  │                                                          │    │
│  │  lida:queue:general   - General tasks                    │    │
│  │  lida:queue:llm       - LLM API calls                    │    │
│  │  lida:queue:analysis  - Analysis tasks                   │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

### 4. Worker System

Background workers process asynchronous tasks.

```
┌─────────────────────────────────────────────────────────────────┐
│                       Worker Pool                                │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │             │
│  │             │  │             │  │             │             │
│  │ Types:      │  │ Types:      │  │ Types:      │             │
│  │ - general   │  │ - llm       │  │ - analysis  │             │
│  │ - compute   │  │ - general   │  │ - io        │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│                          ▼                                      │
│                 ┌─────────────────┐                            │
│                 │   Task Router   │                            │
│                 │                 │                            │
│                 │ Routes tasks to │                            │
│                 │ appropriate     │                            │
│                 │ workers based   │                            │
│                 │ on type         │                            │
│                 └─────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

#### Task Types

| Type | Description | Example |
|------|-------------|---------|
| `general` | Generic tasks | State updates |
| `llm` | LLM API calls | Agent responses |
| `analysis` | Data analysis | Causal inference |
| `compute` | CPU-intensive | Simulations |
| `io` | I/O operations | File exports |

---

### 5. Agent Framework

The core multi-agent system.

```
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Framework                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   Persona Manager                        │    │
│  │                                                          │    │
│  │  169 pre-defined personas with:                          │    │
│  │  - Background & expertise                                │    │
│  │  - Belief systems                                        │    │
│  │  - Communication style                                   │    │
│  │  - Position on AI safety topics                          │    │
│  │                                                          │    │
│  │  Capabilities:                                           │    │
│  │  - Fork personas with modifications                      │    │
│  │  - Track belief evolution                                │    │
│  │  - Assign specific LLM models                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                  Deliberation Engine                     │    │
│  │                                                          │    │
│  │  Modes:                                                  │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │    │
│  │  │  Quorum  │  │  Debate  │  │  Policy Simulation   │   │    │
│  │  │          │  │          │  │                      │   │    │
│  │  │ Multiple │  │ Structured│  │ Game-theoretic      │   │    │
│  │  │ agents   │  │ back-and- │  │ dynamics            │   │    │
│  │  │ deliber- │  │ forth     │  │                      │   │    │
│  │  │ ate on   │  │ debate    │  │ Chip war, AGI race  │   │    │
│  │  │ events   │  │           │  │                      │   │    │
│  │  └──────────┘  └──────────┘  └──────────────────────┘   │    │
│  │                                                          │    │
│  │  Consensus Mechanisms:                                   │    │
│  │  - Voting (majority, supermajority, unanimous)           │    │
│  │  - Weighted by expertise                                 │    │
│  │  - Confidence aggregation (169 strategies)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                   LLM Backend                            │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │    │
│  │  │ OpenRouter  │  │  Anthropic  │  │   OpenAI    │      │    │
│  │  │             │  │             │  │             │      │    │
│  │  │ 100+ models │  │ Claude      │  │ GPT-4       │      │    │
│  │  │ Unified API │  │ family      │  │ family      │      │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘      │    │
│  │                                                          │    │
│  │  ┌─────────────┐  ┌─────────────────────────────┐       │    │
│  │  │    MLX      │  │     Mock/Simulation         │       │    │
│  │  │             │  │                             │       │    │
│  │  │ Local Apple │  │ No API calls for testing   │       │    │
│  │  │ Silicon     │  │                             │       │    │
│  │  └─────────────┘  └─────────────────────────────┘       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

### 6. Advanced Features Architecture

#### Circuit Breaker Pattern

```
                    ┌──────────────────────────┐
                    │      Service Call        │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │    Circuit Breaker       │
                    │                          │
                    │  State: CLOSED/OPEN/     │
                    │         HALF_OPEN        │
                    └────────────┬─────────────┘
                                 │
           ┌─────────────────────┼─────────────────────┐
           │                     │                     │
           ▼                     ▼                     ▼
    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │   CLOSED     │     │    OPEN      │     │  HALF_OPEN   │
    │              │     │              │     │              │
    │ Allow calls  │     │ Reject calls │     │ Allow few    │
    │ Track fails  │     │ Wait timeout │     │ Test recovery│
    └──────────────┘     └──────────────┘     └──────────────┘
```

#### Event Bus

```
┌─────────────────────────────────────────────────────────────────┐
│                         Event Bus                                │
│                                                                  │
│  Publishers                          Subscribers                 │
│  ──────────                          ───────────                 │
│                                                                  │
│  ┌─────────┐                         ┌─────────────────────┐    │
│  │ Service │ ──publish──┐            │ Health Monitor      │    │
│  │ Start   │            │            │ (service.*)         │    │
│  └─────────┘            │            └─────────────────────┘    │
│                         │                                        │
│  ┌─────────┐            ▼            ┌─────────────────────┐    │
│  │ Health  │      ┌──────────┐       │ Metrics Collector   │    │
│  │ Check   │ ────>│  Topic   │──────>│ (metrics.*)         │    │
│  └─────────┘      │  Router  │       └─────────────────────┘    │
│                   └──────────┘                                   │
│  ┌─────────┐            │            ┌─────────────────────┐    │
│  │ Auto    │            │            │ Alert Manager       │    │
│  │ Scaler  │ ──publish──┘            │ (alert.*)           │    │
│  └─────────┘                         └─────────────────────┘    │
│                                                                  │
│  Event Types:                                                    │
│  - service.started, service.stopped, service.failed             │
│  - health.check, health.degraded                                 │
│  - chaos.enabled, chaos.fault_injected                          │
│  - autoscaler.scaled                                             │
│  - snapshot.created, snapshot.restored                          │
└─────────────────────────────────────────────────────────────────┘
```

#### Distributed Tracing

```
Request Flow with Tracing
─────────────────────────

Client Request
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ Trace ID: abc123                                                │
│                                                                 │
│ ┌─────────────────────────────────────────────────────────┐    │
│ │ Span: api-request (root)                                 │    │
│ │ Duration: 1523ms                                         │    │
│ │                                                          │    │
│ │  ┌───────────────────────────────────────────────────┐  │    │
│ │  │ Span: load-personas                                │  │    │
│ │  │ Duration: 45ms                                     │  │    │
│ │  └───────────────────────────────────────────────────┘  │    │
│ │                                                          │    │
│ │  ┌───────────────────────────────────────────────────┐  │    │
│ │  │ Span: deliberation-round-1                         │  │    │
│ │  │ Duration: 456ms                                    │  │    │
│ │  │                                                    │  │    │
│ │  │  ┌─────────────────────────────────────────────┐  │  │    │
│ │  │  │ Span: llm-call (agent-1)                    │  │  │    │
│ │  │  │ Duration: 234ms                             │  │  │    │
│ │  │  └─────────────────────────────────────────────┘  │  │    │
│ │  │                                                    │  │    │
│ │  │  ┌─────────────────────────────────────────────┐  │  │    │
│ │  │  │ Span: llm-call (agent-2)                    │  │  │    │
│ │  │  │ Duration: 198ms                             │  │  │    │
│ │  │  └─────────────────────────────────────────────┘  │  │    │
│ │  └───────────────────────────────────────────────────┘  │    │
│ │                                                          │    │
│ │  ┌───────────────────────────────────────────────────┐  │    │
│ │  │ Span: save-results                                 │  │    │
│ │  │ Duration: 89ms                                     │  │    │
│ │  └───────────────────────────────────────────────────┘  │    │
│ └─────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

---

### 7. Multi-User Architecture

For shared clusters like Mila:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Shared Server (Mila)                          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      User 1 (Linh)                         │  │
│  │  Redis: 6379    API: 2040    Workers: 4                    │  │
│  │  REDIS_URL=redis://localhost:6379                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      User 2 (David)                        │  │
│  │  Redis: 6380    API: 2041    Workers: 4                    │  │
│  │  REDIS_URL=redis://localhost:6380                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                      User 3 (Arthur)                       │  │
│  │  Redis: 6381    API: 2042    Workers: 8                    │  │
│  │  REDIS_URL=redis://localhost:6381                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Port Isolation:                                                 │
│  - Each user has dedicated Redis and API ports                  │
│  - No cross-contamination of state                              │
│  - Independent scaling                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

### 8. Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Data Flow                                    │
│                                                                       │
│  1. Scenario Loading                                                  │
│     scenarios/*.yaml ──> ConfigLoader ──> ExperimentConfig            │
│                                                                       │
│  2. Persona Initialization                                            │
│     persona_pipeline/personas/*.json ──> PersonaManager ──> Agents    │
│                                                                       │
│  3. Deliberation                                                      │
│     Topic ──> Deliberation Engine ──> Agent Responses ──> Consensus   │
│                     │                       │                         │
│                     ▼                       ▼                         │
│               Redis Pub/Sub           LLM Backend                     │
│                                                                       │
│  4. Results                                                           │
│     Consensus ──> Analysis ──> Results JSON ──> Export (LaTeX/PDF)    │
│                                                                       │
│  5. Monitoring                                                        │
│     All stages ──> Event Bus ──> Metrics ──> Health Checks            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Scenario Configuration (YAML)

```yaml
name: ai_safety_debate
version: "1.0"

simulation:
  auto_start_topic: "Should AI development be paused?"
  max_rounds: 10
  timeout_seconds: 600

personas:
  - id: eliezer_yudkowsky
    role: doomer
    model: anthropic/claude-sonnet-4
  - id: marc_andreessen
    role: accelerationist
    model: anthropic/claude-sonnet-4

deliberation:
  mode: debate
  consensus_threshold: 0.7
  voting_method: weighted

output:
  save_transcript: true
  export_format: json
```

### Profile Configuration (JSON)

```json
{
  "dev": {
    "name": "dev",
    "redis_port": 6379,
    "api_port": 2040,
    "workers": 4,
    "live_mode": false,
    "scenario": null,
    "environment": {},
    "services": ["redis", "api", "workers"],
    "description": "Development environment"
  }
}
```

### Cluster Configuration (JSON)

```json
{
  "nodes": {
    "node1": {
      "hostname": "node1",
      "ssh_host": "server1.mila.quebec",
      "redis_port": 6379,
      "api_port": 2040,
      "user": "arthur"
    }
  }
}
```

---

## Security Considerations

### API Security
- Rate limiting on all endpoints
- Optional authentication via API keys
- CORS configuration for web clients

### Distributed Locks
- Automatic lock expiration
- Lock ownership verification before release
- Deadlock prevention via timeouts

### Chaos Engineering Safety
- Disabled by default
- Probability-based (not 100%)
- Can target specific services only
- Easy global disable

---

## Performance Considerations

### Scaling Strategies

| Component | Horizontal | Vertical |
|-----------|------------|----------|
| API Server | Multiple workers | More CPU/RAM |
| Workers | Add more workers | Increase capacity |
| Redis | Redis Cluster | More RAM |

### Bottlenecks

1. **LLM API Calls** - Rate limited by providers
   - Solution: Request pooling, caching

2. **Redis Memory** - Large state accumulation
   - Solution: TTL on keys, periodic cleanup

3. **Worker Starvation** - Too many LLM tasks
   - Solution: Separate queues by task type

---

## Extending LIDA

### Adding New Commands

1. Add command function in `src/cli/main.py`:
```python
def cmd_mycommand(args):
    """My new command."""
    pass
```

2. Add argument parser:
```python
my_parser = subparsers.add_parser("mycommand", help="...")
my_parser.add_argument("--option", ...)
my_parser.set_defaults(func=cmd_mycommand)
```

### Adding New Personas

1. Create JSON file in `persona_pipeline/personas/`:
```json
{
  "id": "new_persona",
  "name": "New Persona",
  "background": "...",
  "beliefs": {...},
  "communication_style": "..."
}
```

### Adding New LLM Backend

1. Implement backend in `src/llm/`:
```python
class MyBackend:
    async def complete(self, prompt: str, **kwargs) -> str:
        pass
```

2. Register in backend factory.

---

*Architecture documentation for LIDA v0.1.0*
