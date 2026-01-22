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
    echo "Usage: $0 <port> [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start all services (redis, api, streaming, workers)"
    echo "  restart   Restart services and workers (leave redis running)"
    echo "  stop      Stop all services including redis"
    echo "  stop-app  Stop services and workers (leave redis running)"
    echo "  status    Show status of all containers"
    echo "  logs      Tail logs from all services"
    echo ""
    echo "Arguments:"
    echo "  port      Required. Redis port number (e.g., 6379, 6380)"
    echo ""
    echo "Environment variables:"
    echo "  API_PORT        API server port (default: 2040)"
    echo "  STREAMING_PORT  Streaming server port (default: 8787)"
    echo "  WORKER_REPLICAS Number of worker containers (default: 1)"
    echo "  NUM_WORKERS     Workers per container (default: 8)"
    echo ""
    echo "Examples:"
    echo "  $0 6379 start           # Start everything on port 6379"
    echo "  $0 6380 start           # Start everything on port 6380 (user 2)"
    echo "  $0 6379 restart         # Restart services, keep redis"
    echo "  $0 6379 stop            # Stop everything"
    echo "  WORKER_REPLICAS=4 $0 6379 start  # Start with 4 worker containers"
    exit 1
}

# Check for required port argument
if [ -z "$1" ]; then
    echo -e "${RED}Error: Port number required${NC}"
    usage
fi

if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo -e "${RED}Error: Port must be a number${NC}"
    usage
fi

PORT=$1
COMMAND=${2:-start}

# Project names based on port
PROJECT_REDIS="lida-redis-${PORT}"
PROJECT_SERVICES="lida-services-${PORT}"
PROJECT_WORKERS="lida-workers-${PORT}"

# Export environment
export REDIS_PORT=$PORT
export API_PORT=${API_PORT:-2040}
export STREAMING_PORT=${STREAMING_PORT:-8787}
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
    if redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; then
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
        if redis-cli -p "$PORT" ping 2>/dev/null | grep -q PONG; then
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
        log_warn "Redis already running on port ${PORT}"
    else
        log_info "Starting Redis on port ${PORT}..."
        docker-compose -f docker-compose.redis.yml -p "$PROJECT_REDIS" up -d
        wait_for_redis
    fi
}

start_services() {
    log_info "Starting services (API: ${API_PORT}, Streaming: ${STREAMING_PORT})..."
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
    echo "=== Redis (port ${PORT}) ==="
    docker-compose -f docker-compose.redis.yml -p "$PROJECT_REDIS" ps 2>/dev/null || echo "Not running"
    echo ""
    echo "=== Services ==="
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

case "$COMMAND" in
    start)
        log_info "Starting LIDA stack on Redis port ${PORT}"
        start_redis
        start_services
        start_workers
        log_info "All services started"
        show_status
        ;;
    restart)
        log_info "Restarting services (keeping Redis on port ${PORT})"
        if ! check_redis_running; then
            log_error "Redis is not running on port ${PORT}. Use 'start' instead."
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
        log_info "Stopping all services on port ${PORT}"
        stop_workers
        stop_services
        stop_redis
        log_info "All services stopped"
        ;;
    stop-app)
        log_info "Stopping services and workers (keeping Redis)"
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
