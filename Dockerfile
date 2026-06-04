# Multi-stage build for the AegeanBench reporter service.
# Final image runs the FastAPI server on port 8200.

FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed for some Python wheels (cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --user --no-cache-dir -r requirements.txt \
    && pip install --user --no-cache-dir fastapi uvicorn httpx pydantic requests

# ----------------------------- runtime -----------------------------

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/root/.local/bin:$PATH

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Application code
COPY aegeanbench /app/aegeanbench
COPY pyproject.toml requirements.txt README.md /app/

# Persisted state directory (mounted from host in docker-compose)
RUN mkdir -p /root/.aegeanbench/worldcup_runs \
             /root/.aegeanbench/sports_cache \
             /root/.aegeanbench/kaggle

EXPOSE 8200

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fs http://localhost:8200/api/v1/health || exit 1

CMD ["uvicorn", "aegeanbench.sports.reporter.server:app", \
     "--host", "0.0.0.0", "--port", "8200", "--workers", "2"]
