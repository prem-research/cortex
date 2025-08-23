# Multi-stage build for production
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY server/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cortex && \
    mkdir -p /app && \
    chown -R cortex:cortex /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/cortex/.local

# Copy application code
COPY --chown=cortex:cortex cortex /app/cortex
COPY --chown=cortex:cortex server /app/server

# Switch to non-root user
USER cortex

# Update PATH
ENV PATH=/home/cortex/.local/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose ports
EXPOSE 8080 50051 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set default environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8080
ENV GRPC_SERVER_PORT=50051
ENV WORKERS=4

# Run the server
CMD ["python", "/app/server/run_server.py"]