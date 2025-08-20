# Multi-stage build for smaller image
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build deps and Python deps
COPY requirements.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc g++ \
       curl \
       ca-certificates \
       libglib2.0-0 libgl1 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Final runtime image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal runtime deps for healthcheck and image processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       curl \
       ca-certificates \
       libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages and scripts from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY src/ ./src/

# Default model dir (PVC will be mounted here in Kubernetes)
# You can override via env MODEL_DIR in the Deployment
# ENV MODEL_DIR=/root/ml-service-models

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
