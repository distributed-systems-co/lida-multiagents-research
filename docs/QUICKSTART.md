# LIDA Quick Start Guide

Get up and running with LIDA in 5 minutes.

## Prerequisites

- Python 3.10+
- Redis server
- API key for LLM provider (OpenRouter recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/lida-multiagents-research.git
cd lida-multiagents-research

# Install dependencies
pip install -e ".[all]"

# Set up API key
export OPENROUTER_API_KEY="sk-or-..."
```

## Verify Installation

```bash
# Check CLI is working
lida --version

# Check system health
lida health check
```

## Start the System

### Option 1: Quick Start (Recommended)

```bash
# Start all services with defaults
lida system start

# Verify everything is running
lida system status
```

### Option 2: Manual Start

```bash
# Terminal 1: Start Redis
redis-server --port 6379

# Terminal 2: Start API server
lida serve --port 2040

# Terminal 3: Start workers (optional)
lida workers --count 4
```

## Run Your First Demo

```bash
# Quick streaming demo
lida demo --type quick

# Interactive AI safety debate
lida debate --interactive
```

## Run a Deliberation

```bash
# Start a deliberation on AI safety
lida deliberate --port 2040 --topic "Should AI development be paused?"

# Watch the output in real-time
# The deliberation will show agent responses as they come in
```

## Common Operations

### Check What's Running

```bash
lida system status
lida health check
```

### Stop Everything

```bash
lida system stop
```

### View Logs

```bash
lida events history
lida events watch  # Real-time
```

## Multi-User Setup (Shared Server)

If multiple people share a server (e.g., Mila cluster):

```bash
# User 1 (default ports)
eval $(lida system env --redis-port 6379 --api-port 2040)
lida system start

# User 2 (different ports)
eval $(lida system env --redis-port 6380 --api-port 2041)
lida system start --redis-port 6380 --api-port 2041
```

## Create a Profile

Save your configuration for easy reuse:

```bash
# Create profile
lida profile create myenv --redis-port 6379 --api-port 2040 --workers 4

# Use it later
eval $(lida profile env myenv)
lida system start
```

## Next Steps

- Read the full [CLI Reference](CLI.md)
- Understand the [Architecture](ARCHITECTURE.md)
- Explore the [Operations Guide](OPERATIONS.md)
- Check out example scenarios in `scenarios/`

## Troubleshooting

### "Port already in use"

```bash
# Check what's using the port
lida system status

# Use a different port
lida system start --api-port 2041
```

### "Cannot connect to Redis"

```bash
# Check if Redis is running
lida health check --name redis

# Start Redis manually
redis-server --port 6379 --daemonize yes
```

### "API not responding"

```bash
# Check API health
lida health check --name api

# Restart the server
lida system stop
lida system start
```

---

*Quick start guide for LIDA v0.1.0*
