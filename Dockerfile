# syntax=docker/dockerfile:1

# ─── Stage 1: builder — isolated venv with serving deps + the app ──────────
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN python -m venv /opt/venv

# CPU-only torch first, from the dedicated index (far smaller than CUDA wheels).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Serving deps only — no dvc / scikit-learn / gdown / test tooling.
COPY requirements-serve.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-serve.txt

# Install app + pulmoscan into the venv (deps already pinned above → --no-deps).
COPY pyproject.toml README.md ./
COPY app/ ./app/
COPY pulmoscan/ ./pulmoscan/
RUN pip install --no-deps .

# ─── Stage 2: runtime — slim image carrying just the venv + model ──────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="PulmoScan AI" \
      org.opencontainers.image.description="Chest CT-scan classifier serving API (FastAPI + PyTorch)" \
      org.opencontainers.image.source="https://github.com/ahammadnafiz/PulmoScan-AI" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENV=production \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH=/app/models/model.pt \
    ENSEMBLE_DIR=/app/models/folds \
    USE_TTA=true \
    WORKERS=1

WORKDIR /app

# Runtime shared libs for Pillow / torch; curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# The Python environment (deps + app + pulmoscan) built in stage 1.
COPY --from=builder /opt/venv /opt/venv

# Bake the single checkpoint if present. models/ always exists (.gitkeep), so a
# model-less build still succeeds — the API then runs degraded until a model is
# mounted. Mount fold checkpoints at $ENSEMBLE_DIR (/app/models/folds) to serve
# the ensemble instead.
COPY models/ ./models/

RUN mkdir -p /app/logs \
    && useradd -m -u 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/v1/health/live || exit 1

# sh -c so $WORKERS expands; exec so uvicorn becomes PID 1 and gets signals cleanly.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
