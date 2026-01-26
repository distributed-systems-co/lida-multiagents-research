# LIDA Operations Guide

Production operations, monitoring, and maintenance procedures.

## Table of Contents

- [Service Management](#service-management)
- [Monitoring](#monitoring)
- [Health Checking](#health-checking)
- [Scaling](#scaling)
- [Fault Tolerance](#fault-tolerance)
- [Backup and Recovery](#backup-and-recovery)
- [Chaos Engineering](#chaos-engineering)
- [Troubleshooting](#troubleshooting)
- [Runbooks](#runbooks)

---

## Service Management

### Starting Services

```bash
# Start all services with health verification
lida system start --wait

# Start with specific configuration
lida system start --redis-port 6379 --api-port 2040 --workers 8

# Start in background
lida system start --background
```

### Stopping Services

```bash
# Graceful shutdown (waits for in-flight requests)
lida system stop

# Force stop all services
lida system stop --force

# Stop specific service
lida orchestrate stop api
```

### Service Orchestration

Services have dependency ordering:

```
redis → api → workers → deliberation
```

Start with dependencies:
```bash
# Start API (automatically starts Redis first)
lida orchestrate start api

# View dependency graph
lida orchestrate list
```

### Rolling Restarts

```bash
# Restart workers one at a time
lida orchestrate restart workers --rolling

# Restart all services in dependency order
lida orchestrate restart --all --rolling
```

---

## Monitoring

### Real-Time Dashboard

```bash
# Terminal UI dashboard
lida dashboard --tui

# Web dashboard
lida dashboard --web --port 8080
```

Dashboard panels:
- Service status (running/stopped/degraded)
- Request throughput
- Error rates
- Resource utilization
- Active deliberations

### Metrics Collection

```bash
# View current metrics
lida metrics show

# Export metrics (Prometheus format)
lida metrics export --format prometheus

# Export to file
lida metrics export --output metrics.json

# View specific metric
lida metrics show --name request_latency
```

Key metrics:
| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `request_latency_p99` | 99th percentile latency | > 5000ms |
| `error_rate` | Errors per second | > 10/s |
| `active_connections` | Current connections | > 1000 |
| `memory_usage_percent` | Memory utilization | > 85% |
| `cpu_usage_percent` | CPU utilization | > 80% |
| `queue_depth` | Pending jobs | > 100 |

### Event Streaming

```bash
# Watch events in real-time
lida events watch

# Filter by type
lida events watch --type error

# View event history
lida events history --limit 100

# Export events
lida events history --format json > events.json
```

### Distributed Tracing

```bash
# View active traces
lida trace list

# Show trace details
lida trace show <trace-id>

# Export traces
lida trace export --format jaeger
```

Trace context propagation:
```python
# Traces are automatically propagated through headers
X-Trace-ID: abc123
X-Span-ID: def456
X-Parent-Span-ID: ghi789
```

---

## Health Checking

### Health Check Commands

```bash
# Check all services
lida health check

# Check specific service
lida health check --name api

# Detailed health report
lida health check --detailed

# JSON output for automation
lida health check --format json
```

### Health Status Codes

| Status | Description | Action |
|--------|-------------|--------|
| `healthy` | All checks passing | None |
| `degraded` | Some checks failing | Investigate |
| `unhealthy` | Critical checks failing | Immediate action |
| `unknown` | Cannot determine status | Check connectivity |

### Custom Health Checks

```bash
# Add custom check
lida health add-check --name "database" \
  --command "pg_isready -h localhost" \
  --interval 30

# Remove check
lida health remove-check --name "database"

# List all checks
lida health list-checks
```

### Readiness vs Liveness

```bash
# Readiness probe (can accept traffic?)
lida health ready

# Liveness probe (is process alive?)
lida health live

# Both (for Kubernetes)
lida health probe --type all
```

---

## Scaling

### Manual Scaling

```bash
# Scale workers
lida autoscale set --service workers --replicas 8

# Scale API instances
lida autoscale set --service api --replicas 4
```

### Auto-Scaling

```bash
# Enable auto-scaling
lida autoscale enable --service workers

# Configure scaling policy
lida autoscale configure --service workers \
  --min 2 \
  --max 16 \
  --target-cpu 70

# View current scale
lida autoscale status

# Disable auto-scaling
lida autoscale disable --service workers
```

Scaling policies:
```yaml
# Example policy configuration
workers:
  min_replicas: 2
  max_replicas: 16
  scale_up_threshold: 80    # CPU %
  scale_down_threshold: 30  # CPU %
  cooldown_period: 300      # seconds
```

### Resource Limits

```bash
# Set memory limit
lida orchestrate config api --memory-limit 4G

# Set CPU limit
lida orchestrate config workers --cpu-limit 2.0
```

---

## Fault Tolerance

### Circuit Breakers

```bash
# View circuit status
lida circuit status

# Open circuit (stop traffic)
lida circuit open --name api-backend

# Close circuit (resume traffic)
lida circuit close --name api-backend

# Reset circuit (clear error counts)
lida circuit reset --name api-backend
```

Circuit states:
```
CLOSED → (failures exceed threshold) → OPEN
                                         ↓
                                    (timeout)
                                         ↓
                                    HALF_OPEN
                                         ↓
              (success) ← ─ ─ ─ ─ ─ ─ ─ ↓ ─ ─ ─ ─ ─ ─ → (failure)
                  ↓                                          ↓
               CLOSED                                      OPEN
```

Configuration:
```bash
# Configure circuit breaker
lida circuit configure --name api-backend \
  --failure-threshold 5 \
  --recovery-timeout 30 \
  --half-open-requests 3
```

### Rate Limiting

```bash
# View rate limit status
lida metrics show --name rate_limit

# Configure rate limits (in code)
# Token bucket: 100 requests, refill 10/second
# Sliding window: 1000 requests per minute
```

### Retry Policies

Built-in retry with exponential backoff:
```python
# Automatic retries with:
# - Initial delay: 1s
# - Max delay: 60s
# - Backoff multiplier: 2
# - Jitter: ±10%
```

---

## Backup and Recovery

### Snapshots

```bash
# Create snapshot
lida snapshot create --name "pre-upgrade-$(date +%Y%m%d)"

# List snapshots
lida snapshot list

# Restore from snapshot
lida snapshot restore --name "pre-upgrade-20240115"

# Delete old snapshots
lida snapshot delete --older-than 30d
```

Snapshot contents:
- Redis data (RDB dump)
- Configuration files
- Service state
- Profile data

### Automated Backups

```bash
# Schedule daily backups
lida schedule add backup-daily \
  --cron "0 2 * * *" \
  --command "lida snapshot create --name backup-\$(date +%Y%m%d)"

# Verify backup schedule
lida schedule list
```

### Disaster Recovery

1. **Stop all services**
   ```bash
   lida system stop --force
   ```

2. **Restore from snapshot**
   ```bash
   lida snapshot restore --name <snapshot-name>
   ```

3. **Verify data integrity**
   ```bash
   lida health check --detailed
   ```

4. **Restart services**
   ```bash
   lida system start --wait
   ```

---

## Chaos Engineering

### Fault Injection

```bash
# Inject latency
lida chaos inject latency --target api --delay 500ms --duration 60s

# Inject errors
lida chaos inject error --target workers --rate 10 --duration 30s

# Kill random instances
lida chaos inject kill --target workers --percent 25

# Network partition simulation
lida chaos inject partition --between api,redis --duration 120s
```

### Chaos Experiments

```bash
# View active experiments
lida chaos status

# Stop all chaos
lida chaos stop

# Run chaos game day
lida chaos gameday --scenario network-partition
```

### Safety Controls

```bash
# Set blast radius limit
lida chaos configure --max-affected 50

# Enable auto-rollback
lida chaos configure --auto-rollback true

# Set experiment timeout
lida chaos configure --max-duration 300
```

---

## Troubleshooting

### Common Issues

#### Port Already in Use

```bash
# Find process using port
lsof -i :2040

# Use different port
lida system start --api-port 2041

# Or kill existing process
kill -9 <pid>
```

#### Redis Connection Failed

```bash
# Check Redis status
lida health check --name redis

# Verify Redis is running
redis-cli ping

# Check Redis logs
tail -f /var/log/redis/redis-server.log

# Restart Redis
lida orchestrate restart redis
```

#### High Memory Usage

```bash
# Check memory usage
lida metrics show --name memory_usage

# View per-service memory
lida orchestrate status --detailed

# Force garbage collection
lida system gc

# Scale down if needed
lida autoscale set --service workers --replicas 2
```

#### High Latency

```bash
# Check latency metrics
lida metrics show --name request_latency

# View traces for slow requests
lida trace list --min-duration 5000

# Check circuit breaker status
lida circuit status

# Enable rate limiting
lida orchestrate config api --rate-limit 100
```

#### Service Won't Start

```bash
# Check dependencies
lida orchestrate deps api

# View logs
lida events history --service api --level error

# Check configuration
lida profile show

# Validate configuration
lida system validate
```

### Debug Mode

```bash
# Start with debug logging
LIDA_DEBUG=1 lida system start

# Enable verbose output
lida --verbose <command>

# Trace specific request
lida trace create --name debug-request
```

### Log Analysis

```bash
# View recent errors
lida events history --level error --limit 50

# Search logs
lida events search "connection refused"

# Export for analysis
lida events history --format json > debug.json
```

---

## Runbooks

### Runbook: High Error Rate

**Symptoms**: Error rate > 10/s, alerts firing

**Steps**:
1. Check error types
   ```bash
   lida events history --level error --limit 20
   ```

2. Identify affected service
   ```bash
   lida health check --detailed
   ```

3. Check circuit breakers
   ```bash
   lida circuit status
   ```

4. If downstream failure, open circuit
   ```bash
   lida circuit open --name <service>
   ```

5. Scale up healthy services
   ```bash
   lida autoscale set --service workers --replicas 8
   ```

6. Investigate root cause via traces
   ```bash
   lida trace list --status error --limit 10
   ```

### Runbook: Service Degradation

**Symptoms**: Latency increasing, timeouts occurring

**Steps**:
1. Check resource utilization
   ```bash
   lida metrics show
   ```

2. If CPU > 80%, scale up
   ```bash
   lida autoscale set --service <name> --replicas +2
   ```

3. If memory > 85%, restart service
   ```bash
   lida orchestrate restart <service> --rolling
   ```

4. Enable rate limiting if overwhelmed
   ```bash
   lida orchestrate config api --rate-limit 50
   ```

### Runbook: Complete Outage

**Symptoms**: All services down, no connectivity

**Steps**:
1. Check infrastructure
   ```bash
   ping localhost
   redis-cli ping
   ```

2. Force stop everything
   ```bash
   lida system stop --force
   ```

3. Clear any stuck locks
   ```bash
   lida lock release --all
   ```

4. Start core services first
   ```bash
   lida orchestrate start redis
   lida orchestrate start api
   ```

5. Verify health
   ```bash
   lida health check
   ```

6. Start remaining services
   ```bash
   lida system start
   ```

### Runbook: Scheduled Maintenance

**Steps**:
1. Create snapshot
   ```bash
   lida snapshot create --name "pre-maintenance-$(date +%Y%m%d)"
   ```

2. Notify users (if applicable)

3. Enable maintenance mode
   ```bash
   lida system maintenance --enable
   ```

4. Perform maintenance

5. Verify health
   ```bash
   lida health check --detailed
   ```

6. Disable maintenance mode
   ```bash
   lida system maintenance --disable
   ```

7. Monitor for issues
   ```bash
   lida events watch --duration 300
   ```

---

## Multi-User Operations (Mila Cluster)

### User Isolation

Each user should have isolated resources:

```bash
# User 1: David
export LIDA_USER=david
eval $(lida system env --redis-port 6379 --api-port 2040)

# User 2: Linh
export LIDA_USER=linh
eval $(lida system env --redis-port 6380 --api-port 2041)
```

### Shared Resource Management

```bash
# View all user instances
lida cluster list

# Check resource usage per user
lida cluster usage

# Coordinate maintenance
lida cluster broadcast "Maintenance in 30 minutes"
```

### Port Allocation

| User | Redis Port | API Port | Worker Ports |
|------|------------|----------|--------------|
| david | 6379 | 2040 | 3000-3003 |
| linh | 6380 | 2041 | 3004-3007 |
| guest | 6381 | 2042 | 3008-3011 |

---

*Operations guide for LIDA v0.1.0*
