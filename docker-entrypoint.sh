#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# LIDA Multi-Agent System - Docker Entrypoint
# Flexible startup script for different run modes
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  LIDA Multi-Agent Deliberation System${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Default values
PORT=${PORT:-12345}
HOST=${HOST:-0.0.0.0}
WORKERS=${WORKERS:-1}
LOG_LEVEL=${LOG_LEVEL:-info}
REDIS_URL=${REDIS_URL:-redis://localhost:6379}

# Wait for Redis if configured
wait_for_redis() {
    if [ -n "$REDIS_URL" ]; then
        echo -e "${YELLOW}Waiting for Redis...${NC}"

        # Extract host and port from URL
        REDIS_HOST=$(echo $REDIS_URL | sed -e 's|redis://||' -e 's|:.*||')
        REDIS_PORT=$(echo $REDIS_URL | sed -e 's|redis://[^:]*:||' -e 's|/.*||')
        REDIS_PORT=${REDIS_PORT:-6379}

        MAX_RETRIES=30
        RETRY_COUNT=0

        while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
            if nc -z $REDIS_HOST $REDIS_PORT 2>/dev/null; then
                echo -e "${GREEN}Redis is available at $REDIS_HOST:$REDIS_PORT${NC}"
                return 0
            fi
            RETRY_COUNT=$((RETRY_COUNT + 1))
            echo "Waiting for Redis... ($RETRY_COUNT/$MAX_RETRIES)"
            sleep 1
        done

        echo -e "${RED}Warning: Could not connect to Redis, continuing anyway...${NC}"
    fi
}

# Print configuration
print_config() {
    echo -e "${GREEN}Configuration:${NC}"
    echo "  PORT: $PORT"
    echo "  HOST: $HOST"
    echo "  WORKERS: $WORKERS"
    echo "  LOG_LEVEL: $LOG_LEVEL"
    echo "  REDIS_URL: ${REDIS_URL:-not set}"
    echo "  OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+***set***}"
    echo ""
}

# Main command handler
case "${1:-server}" in
    server)
        echo -e "${GREEN}Starting Swarm Server...${NC}"
        print_config
        wait_for_redis
        exec uvicorn run_swarm_server:app \
            --host "$HOST" \
            --port "$PORT" \
            --workers "$WORKERS" \
            --log-level "$LOG_LEVEL" \
            --access-log
        ;;

    server-live)
        echo -e "${GREEN}Starting Swarm Server (LIVE MODE)...${NC}"
        print_config
        wait_for_redis
        exec python run_swarm_server.py --live --port "$PORT"
        ;;

    server-advanced)
        echo -e "${GREEN}Starting Advanced Swarm Server...${NC}"
        echo -e "${BLUE}Full LIDA Architecture with Redis Messaging${NC}"
        print_config
        wait_for_redis
        AGENTS=${SWARM_AGENTS:-6}
        exec python run_swarm_server.py --advanced --agents "$AGENTS" --port "$PORT"
        ;;

    server-advanced-live)
        echo -e "${GREEN}Starting Advanced Swarm Server (LIVE MODE)...${NC}"
        echo -e "${BLUE}Full LIDA Architecture with LLM Integration${NC}"
        print_config
        wait_for_redis
        AGENTS=${SWARM_AGENTS:-6}
        exec python run_swarm_server.py --advanced --live --agents "$AGENTS" --port "$PORT"
        ;;

    experiment)
        echo -e "${GREEN}Running Persuasion Experiment...${NC}"
        print_config
        exec python run_persuasion_experiment.py "${@:2}"
        ;;

    experiment-live)
        echo -e "${GREEN}Running Persuasion Experiment (LIVE MODE)...${NC}"
        print_config
        exec python run_persuasion_experiment.py --live "${@:2}"
        ;;

    dashboard)
        echo -e "${GREEN}Starting Swarm Dashboard...${NC}"
        print_config
        wait_for_redis
        exec python run_swarm_dashboard.py --port "$PORT"
        ;;

    workers)
        echo -e "${GREEN}Starting Worker Pool...${NC}"
        NUM_WORKERS=${NUM_WORKERS:-4}
        WORKER_CAPACITY=${WORKER_CAPACITY:-5}
        echo "  NUM_WORKERS: $NUM_WORKERS"
        echo "  WORKER_CAPACITY: $WORKER_CAPACITY"
        echo "  REDIS_URL: $REDIS_URL"
        echo ""
        wait_for_redis
        exec python run_workers.py \
            --num-workers "$NUM_WORKERS" \
            --capacity "$WORKER_CAPACITY" \
            --redis-url "$REDIS_URL"
        ;;

    export-personas)
        echo -e "${GREEN}Exporting Personas to YAML...${NC}"
        VERSION=${2:-v2}
        exec python -m src.manipulation.persona_yaml export --version "$VERSION"
        ;;

    validate-personas)
        echo -e "${GREEN}Validating Personas...${NC}"
        VERSION=${2:-v2}
        exec python -m src.manipulation.persona_yaml validate --version "$VERSION"
        ;;

    shell)
        echo -e "${GREEN}Starting interactive shell...${NC}"
        exec /bin/bash
        ;;

    python)
        echo -e "${GREEN}Starting Python REPL...${NC}"
        exec python "${@:2}"
        ;;

    test)
        echo -e "${GREEN}Running tests...${NC}"
        exec pytest tests/ -v "${@:2}"
        ;;

    help|--help|-h)
        echo "Usage: docker run lida-multiagents [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  server            Start the web server (default)"
        echo "  server-live       Start server with live LLM calls"
        echo "  server-advanced   Start advanced server with full LIDA architecture"
        echo "  server-advanced-live  Advanced server with LLM integration"
        echo "  workers           Start background worker pool"
        echo "  experiment        Run persuasion experiments"
        echo "  experiment-live   Run experiments with live LLM calls"
        echo "  dashboard         Start the swarm dashboard"
        echo "  export-personas   Export personas to YAML (v1|v2)"
        echo "  validate-personas Validate persona YAML files"
        echo "  shell             Start interactive bash shell"
        echo "  python [args]     Run Python with arguments"
        echo "  test [args]       Run pytest with arguments"
        echo "  help              Show this help message"
        echo ""
        echo "Environment Variables:"
        echo "  PORT              Server port (default: 12345)"
        echo "  HOST              Server host (default: 0.0.0.0)"
        echo "  WORKERS           Number of uvicorn workers (default: 1)"
        echo "  SWARM_AGENTS      Number of persona agents for advanced mode (default: 6)"
        echo "  NUM_WORKERS       Number of worker agents (default: 4)"
        echo "  WORKER_CAPACITY   Tasks per worker (default: 5)"
        echo "  LOG_LEVEL         Log level (default: info)"
        echo "  REDIS_URL         Redis connection URL"
        echo "  OPENROUTER_API_KEY  API key for OpenRouter (access to Claude, GPT, etc)"
        exit 0
        ;;

    *)
        # Pass through any other command
        exec "$@"
        ;;
esac
