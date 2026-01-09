# ═══════════════════════════════════════════════════════════════════════════════
# LIDA Multi-Agent System - Production Dockerfile
# For deployment on Maple and other Linux servers
# ═══════════════════════════════════════════════════════════════════════════════

# Build stage
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════════════════════════════════════
# Production stage
FROM python:3.12-slim as production

# Labels
LABEL maintainer="Arthur Collé"
LABEL description="LIDA Multi-Agent Deliberation System"
LABEL version="1.0.0"

# Create non-root user for security
RUN groupadd -r lida && useradd -r -g lida lida

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=lida:lida . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/experiment_results && \
    chown -R lida:lida /app

# Copy entrypoint script
COPY --chown=lida:lida docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER lida

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PORT=12345 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Default entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Default command (can be overridden)
CMD ["server"]
