#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    echo "Usage: $0 <redis_port> <api_port> [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start all services (redis, api, workers)"
    echo "  restart   Restart services and workers (leave redis running)"
    echo "  stop      Stop all services including redis"
    echo "  stop-app  Stop services and workers (leave redis running)"
    echo "  status    Show status of all containers"
    echo "  logs      Tail logs from all services"
    echo ""
    echo "Arguments:"
    echo "  redis_port  Required. Redis port number (e.g., 6379, 6380)"
    echo "  api_port    Required. API server port (e.g., 2040, 2041)"
    echo ""
    echo "Environment variables:"
    echo "  WORKER_REPLICAS Number of worker containers (default: 1)"
    echo "  NUM_WORKERS     Workers per container (default: 8)"
    echo ""
    echo "Examples:"
    echo "  $0 6379 2040 start      # Start everything (Redis 6379, API 2040)"
    echo "  $0 6380 2041 start      # Start everything (Redis 6380, API 2041) - user 2"
    echo "  $0 6379 2040 restart    # Restart services, keep redis"
    echo "  $0 6379 2040 stop       # Stop everything"
    echo "  WORKER_REPLICAS=4 $0 6379 2040 start  # Start with 4 worker containers"
    exit 1
}

# Check for required arguments
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "${RED}Error: Redis port and API port required${NC}"
    usage
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Redis port must be a number${NC}"
    usage
fi

if ! [[ "$2" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: API port must be a number${NC}"
    usage
fi

REDIS_PORT_ARG=$1
API_PORT_ARG=$2
COMMAND=${3:-start}

# Project names based on ports
PROJECT_REDIS="lida-redis-${REDIS_PORT_ARG}"
PROJECT_SERVICES="lida-services-${REDIS_PORT_ARG}-${API_PORT_ARG}"
PROJECT_WORKERS="lida-workers-${REDIS_PORT_ARG}-${API_PORT_ARG}"

# Export environment
export REDIS_PORT=$REDIS_PORT_ARG
export API_PORT=$API_PORT_ARG
export WORKER_REPLICAS=${WORKER_REPLICAS:-1}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_redis_running() {
    # Check if redis is responding on the port
    if redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
        return 0
    fi
    # Fallback: check for container with project name
    if docker ps --format '{{.Names}}' | grep -qE "${PROJECT_REDIS}.*redis"; then
        return 0
    fi
    return 1
}

wait_for_redis() {
    log_info "Waiting for Redis to be healthy..."
    local max_attempts=30
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        # Try connecting directly to the port
        if redis-cli -p "$REDIS_PORT" ping 2>/dev/null | grep -q PONG; then
            log_info "Redis is ready"
            return 0
        fi
        # Fallback: try docker exec with various naming conventions
        local container=$(docker ps --format '{{.Names}}' | grep -E "${PROJECT_REDIS}.*redis" | head -1)
        if [ -n "$container" ] && docker exec "$container" redis-cli ping 2>/dev/null | grep -q PONG; then
            log_info "Redis is ready"
            return 0
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo ""
    log_error "Redis failed to become healthy"
    return 1
}

start_redis() {
    if check_redis_running; then
        log_warn "Redis already running on port ${REDIS_PORT}"
    else
        log_info "Starting Redis on port ${REDIS_PORT}..."
        docker-compose -f docker-compose.redis.yml -p "$PROJECT_REDIS" up -d
        wait_for_redis
    fi
}

start_services() {
    log_info "Starting API server on port ${API_PORT}..."
    docker-compose -f docker-compose.services.yml -p "$PROJECT_SERVICES" up -d
}

start_workers() {
    log_info "Starting workers (replicas: ${WORKER_REPLICAS})..."
    docker-compose -f docker-compose.workers.yml -p "$PROJECT_WORKERS" up -d
}

stop_redis() {
    log_info "Stopping Redis..."
    docker-compose -f docker-compose.redis.yml -p "$PROJECT_REDIS" down
}

stop_services() {
    log_info "Stopping services..."
    docker-compose -f docker-compose.services.yml -p "$PROJECT_SERVICES" down
}

stop_workers() {
    log_info "Stopping workers..."
    docker-compose -f docker-compose.workers.yml -p "$PROJECT_WORKERS" down
}

show_status() {
    echo ""
    echo "=== Redis (port ${REDIS_PORT}) ==="
    docker-compose -f docker-compose.redis.yml -p "$PROJECT_REDIS" ps 2>/dev/null || echo "Not running"
    echo ""
    echo "=== API (port ${API_PORT}) ==="
    docker-compose -f docker-compose.services.yml -p "$PROJECT_SERVICES" ps 2>/dev/null || echo "Not running"
    echo ""
    echo "=== Workers ==="
    docker-compose -f docker-compose.workers.yml -p "$PROJECT_WORKERS" ps 2>/dev/null || echo "Not running"
}

show_logs() {
    docker-compose -f docker-compose.redis.yml -p "$PROJECT_REDIS" logs -f &
    docker-compose -f docker-compose.services.yml -p "$PROJECT_SERVICES" logs -f &
    docker-compose -f docker-compose.workers.yml -p "$PROJECT_WORKERS" logs -f &
    wait
}

# Ensure logs directory exists with local user ownership (before Docker creates it as root)
mkdir -p logs

case "$COMMAND" in
    start)
        log_info "Starting LIDA stack (Redis: ${REDIS_PORT}, API: ${API_PORT})"
        start_redis
        start_services
        start_workers
        log_info "All services started"
        show_status
        ;;
    restart)
        log_info "Restarting services (keeping Redis on port ${REDIS_PORT})"
        if ! check_redis_running; then
            log_error "Redis is not running on port ${REDIS_PORT}. Use 'start' instead."
            exit 1
        fi
        stop_workers
        stop_services
        start_services
        start_workers
        log_info "Services restarted"
        show_status
        ;;
    stop)
        log_info "Stopping all services (Redis: ${REDIS_PORT}, API: ${API_PORT})"
        stop_workers
        stop_services
        stop_redis
        log_info "All services stopped"
        ;;
    stop-app)
        log_info "Stopping services and workers (keeping Redis on ${REDIS_PORT})"
        stop_workers
        stop_services
        log_info "Services stopped, Redis still running"
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    *)
        echo -e "${RED}Unknown command: ${COMMAND}${NC}"
        usage
        ;;
esac
