# LIDA Multi-Agent Research

Multi-agent system with Redis pub/sub messaging, inbound/outbound mailbox dynamics, and demiurge orchestration.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose
- (Optional) OpenRouter API key for LLM features

## Setup Guide

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/lida-multiagents-research.git
cd lida-multiagents-research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"      # Development (includes testing, linting)
pip install -e ".[llm]"      # LLM support (Anthropic, OpenAI)
pip install -e ".[dev,llm]"  # Both

# Start Redis
docker-compose up -d redis

# Run simulation
python run_simulation.py
```

### Environment Variables

Create a `.env` file in the project root:

```bash
REDIS_URL=redis://localhost:6379
OPENROUTER_API_KEY=your_api_key_here  # Optional, for LLM features
```

## Docker

### Build and Run with Docker

```bash
# Build the image
docker build -t lida-multiagents .

# Run the container
docker run -p 12345:12345 -e REDIS_URL=redis://host.docker.internal:6379 lida-multiagents
```

### Docker Compose (Recommended)

Run the full stack with Redis and monitoring:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop all services
docker-compose down
```

Available services:
| Service | Port | Description |
|---------|------|-------------|
| app | 12345 | Main FastAPI application |
| redis | 6379 | Redis message broker |
| redis-commander | 8081 | Redis web UI for debugging |

### Development with Docker

Mount local code for hot-reloading:

```bash
docker-compose up -d
# Code changes are reflected automatically via volume mount
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_messaging.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Message Broker                            │
│                    (Redis Pub/Sub + Streams)                     │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
              ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│     Demiurge Agent      │         │    Persona Agent (N)    │
│  ┌───────────────────┐  │         │  ┌───────────────────┐  │
│  │  Inbound Mailbox  │◄─┼─────────┼──│  Inbound Mailbox  │  │
│  │  (priority queue) │  │         │  │  (priority queue) │  │
│  └───────────────────┘  │         │  └───────────────────┘  │
│  ┌───────────────────┐  │         │  ┌───────────────────┐  │
│  │ Outbound Mailbox  │──┼─────────┼─▶│ Outbound Mailbox  │  │
│  │  (rate limited)   │  │         │  │  (rate limited)   │  │
│  └───────────────────┘  │         │  └───────────────────┘  │
│  ┌───────────────────┐  │         │  ┌───────────────────┐  │
│  │   World State     │  │         │  │  Knowledge Base   │  │
│  │   (S,L,M,O,A,V,R) │  │         │  │   + Beliefs       │  │
│  └───────────────────┘  │         │  └───────────────────┘  │
└─────────────────────────┘         └─────────────────────────┘
```

## Components

### Messaging System (`src/messaging/`)

- **Message**: Core message type with priority, TTL, correlation tracking
- **InboundMailbox**: Priority queue with filtering, backpressure, dead letters
- **OutboundMailbox**: Rate-limited sending with batching and ACK tracking
- **MessageBroker**: Redis-backed routing, pub/sub, persistence

### Agents (`src/agents/`)

- **DemiurgeAgent**: Orchestrator implementing the craftsman-intelligence pattern
- **PersonaAgent**: Domain expert agents instantiated from 920 system prompts

## Message Types

| Type | Description |
|------|-------------|
| DIRECT | Point-to-point |
| BROADCAST | To all agents |
| MULTICAST | To topic subscribers |
| REQUEST/RESPONSE | RPC-style |
| PROPOSE/VOTE/COMMIT | Consensus |
| SPAWN/TERMINATE | Lifecycle |

## Usage

### Basic Agent Communication

```python
# Send direct message
await agent.send(
    recipient_id="other_agent",
    msg_type=MessageType.DIRECT,
    payload={"key": "value"},
)

# Broadcast to all
await agent.broadcast(
    MessageType.BROADCAST,
    {"announcement": "Hello world"},
)

# Request/Response
response = await agent.request(
    "target_agent",
    {"action": "query", "query": "status"},
    timeout=30.0,
)

# Pub/Sub
await agent.subscribe("topic:events")
await agent.publish("topic:events", {"event": "something_happened"})
```

### Spawning Agents

```python
registry = AgentRegistry(broker)
registry.register_type("demiurge", DemiurgeAgent)
registry.register_type("persona", PersonaAgent)

# Spawn with config
agent = await registry.spawn(
    "persona",
    AgentConfig(agent_id="expert_1"),
    persona_prompt="You are an expert in...",
)
```

## File Structure

```
lida-multiagents-research/
├── src/
│   ├── messaging/
│   │   ├── __init__.py
│   │   ├── messages.py      # Message, Envelope, MessageType
│   │   ├── mailbox.py       # Inbound/Outbound mailboxes
│   │   ├── broker.py        # Redis message broker
│   │   └── agent.py         # Base Agent class, Registry
│   └── agents/
│       ├── __init__.py
│       ├── demiurge.py      # Demiurge orchestrator
│       └── persona.py       # Persona agents
├── demiurge.prompts/        # Demiurge system components
├── populations.prompts/     # 920 persona prompts
├── specs/                   # Schemas and specs
├── tools/                   # Validation tools
├── tests/                   # Test suite
├── run_simulation.py        # Main runner
├── docker-compose.yml       # Redis setup
└── requirements.txt
```

## License

MIT
