# ─────────────────────────────────────────────────────────────
# Dockerfile — Smart Task Management Agent
# Multi-stage build: builder (deps) → runtime (code only)
#
# Build : docker build -t smart-task-agent .
# Run   : docker run -p 8000:8000 --env-file .env smart-task-agent
# Verify: curl http://localhost:8000/health
# Docs  : http://localhost:8000/docs
# ─────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first so Docker caches this layer separately from code
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy only installed packages from builder (no build tools in final image)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY app/        ./app/
COPY evaluation/ ./evaluation/

# Create writable runtime directories owned by appuser
RUN mkdir -p app/checkpoints screenshots \
 && chown -R appuser:appuser /app

USER appuser

# ── Environment defaults (override with --env-file .env) ──────
ENV APP_ENV=production
ENV LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Docker HEALTHCHECK — used by Render, HuggingFace, k8s liveness probes
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
